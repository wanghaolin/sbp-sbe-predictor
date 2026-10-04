"""NAM-TabR hybrid model (latest framework).

Components:
    (i)   Linear baseline: a global linear layer over all features plus a
          dedicated linear pathway for the prespecified core features.
    (ii)  NAM: one small MLP per feature; zero-meaned per-feature outputs are
          summed into additive, interpretable contributions.
    (iii) Core-feature interaction network: a compact MLP over the core
          features capturing their nonlinear interactions.
    (iv)  TabR (Siamese retrieval): dynamic pools of positive AND negative
          training anchors; each sample is scored by core-weighted cosine
          similarity to its nearest neighbours in each pool, and the
          positive-versus-negative similarity difference becomes a bonus logit.

Rather than an uncertainty gate, each nonlinear component is combined through
its own learnable gate (alpha / beta / gamma = sigmoid of a trainable scalar),
initialised near zero so the model defaults to the interpretable linear
prediction and activates nonlinear terms only when they reduce the loss.

Training uses focal loss (with label smoothing) and class-balanced sampling.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance, with optional label smoothing."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets_smooth * probs + (1 - targets_smooth) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_w = targets_smooth * self.alpha + (1 - targets_smooth) * (1 - self.alpha)

        if self.pos_weight is not None:
            pw = targets * (self.pos_weight - 1) + 1
            alpha_w = alpha_w * pw

        return (alpha_w * focal_weight * bce).mean()


class NAM(nn.Module):
    """Neural Additive Model: one small MLP per feature, zero-meaned and summed.

    Produces an additive (interpretable) logit contribution per feature.
    """

    def __init__(self, num_features: int, hidden_dim: int = 48,
                 dropout: float = 0.35, feature_dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.feature_dropout = feature_dropout

        self.feature_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SELU(),
                nn.AlphaDropout(dropout),
                nn.Linear(hidden_dim // 2, 1, bias=False),
            )
            for _ in range(num_features)
        ])

        for net in self.feature_networks:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        """x: [batch, num_features] -> additive logit [batch, 1]."""
        if is_train and self.feature_dropout > 0:
            mask = torch.bernoulli(
                torch.full((x.size(0), self.num_features), 1 - self.feature_dropout,
                           device=x.device)
            )
            x = x * mask

        outs = []
        for i, net in enumerate(self.feature_networks):
            out = net(x[:, i:i + 1])
            out = out - out.mean(dim=0, keepdim=True)
            outs.append(out)
        return torch.cat(outs, dim=1).sum(dim=1, keepdim=True)


class TabRSiamese(nn.Module):
    """TabR retrieval with both positive and negative anchor pools.

    Scores each sample by the similarity difference between its nearest
    positive and negative neighbours (core features up-weighted).
    """

    def __init__(self, num_features: int, core_indices: List[int],
                 hidden_dim: int = 32, core_weight: float = 8.0,
                 pool_size: int = 300, top_k: int = 10, use_cosine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.pool_size = pool_size
        self.top_k = top_k
        self.use_cosine = use_cosine

        self.register_buffer('feature_weights', torch.ones(num_features))
        for idx in core_indices:
            self.feature_weights[idx] = core_weight

        self.register_buffer('pos_pool', torch.zeros(pool_size, num_features))
        self.register_buffer('pos_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('pos_filled', torch.zeros(1, dtype=torch.bool))

        self.register_buffer('neg_pool', torch.zeros(pool_size, num_features))
        self.register_buffer('neg_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('neg_filled', torch.zeros(1, dtype=torch.bool))

        self.retrieval_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.retrieval_net[-1].weight, 0.01)
        nn.init.constant_(self.retrieval_net[-1].bias, 0.0)

    def update_pool(self, x: torch.Tensor, y: torch.Tensor):
        pos_x = x[(y == 1).squeeze()]
        neg_x = x[(y == 0).squeeze()]

        if pos_x.size(0) > 0:
            n_new = min(pos_x.size(0), self.pool_size)
            start = self.pos_ptr.item() % self.pool_size
            for i in range(n_new):
                self.pos_pool[(start + i) % self.pool_size] = pos_x[i]
            self.pos_ptr += n_new
            if not self.pos_filled and self.pos_ptr >= self.pool_size:
                self.pos_filled.fill_(True)

        if neg_x.size(0) > 0:
            n_new = min(neg_x.size(0), self.pool_size)
            start = self.neg_ptr.item() % self.pool_size
            for i in range(n_new):
                self.neg_pool[(start + i) % self.pool_size] = neg_x[i]
            self.neg_ptr += n_new
            if not self.neg_filled and self.neg_ptr >= self.pool_size:
                self.neg_filled.fill_(True)

    def _get_similarity(self, x, pool, pool_ptr, pool_filled):
        if not pool_filled and pool_ptr <= 0:
            return None
        ps = min(pool_ptr.item(), self.pool_size)
        anchors = pool[:ps]
        wx = x * self.feature_weights
        wa = anchors * self.feature_weights
        if self.use_cosine:
            sim = F.normalize(wx, dim=-1) @ F.normalize(wa, dim=-1).T
        else:
            sim = torch.exp(-torch.cdist(wx, wa).pow(2) / (2 * self.num_features))
        tk = min(self.top_k, sim.size(-1))
        return sim.topk(tk, dim=-1)[0].mean(dim=-1, keepdim=True)

    def forward(self, x, candidate_x=None, candidate_y=None, is_train=True):
        """x: [batch, num_features] -> retrieval bonus logit [batch, 1]."""
        bonus = torch.zeros(x.size(0), 1, device=x.device)
        if is_train and candidate_x is not None and candidate_y is not None:
            self.update_pool(candidate_x, candidate_y)

        pos_sim = self._get_similarity(x, self.pos_pool, self.pos_ptr, self.pos_filled)
        neg_sim = self._get_similarity(x, self.neg_pool, self.neg_ptr, self.neg_filled)

        if pos_sim is None or neg_sim is None:
            return bonus

        sim_diff = pos_sim - neg_sim
        features = torch.cat([pos_sim, sim_diff], dim=-1)
        return self.retrieval_net(features)


class LRTabRModel(nn.Module):
    """Linear backbone + core-feature linear term + TabR Siamese retrieval.

    Optionally adds a gated NAM (additive) term. The NAM gate is initialised
    near zero (``nam_gate_init`` very negative) so the model starts equivalent
    to the linear+TabR baseline and only grows the NAM contribution if it helps.
    """

    def __init__(self, num_features, core_indices, hidden_dim=32, dropout=0.2,
                 core_weight=8.0, tabr_pool=300, tabr_k=10, tabr_cos=True,
                 alpha_init=0.1, use_nam=False, nam_hidden=48, nam_dropout=0.35,
                 nam_feature_dropout=0.1, nam_gate_init=-4.0):
        super().__init__()
        self.num_features = num_features
        self.core_indices = core_indices
        self.use_nam = use_nam

        self.lr = nn.Linear(num_features, 1)
        nn.init.xavier_uniform_(self.lr.weight, gain=0.5)
        nn.init.zeros_(self.lr.bias)

        self.core_lr = nn.Linear(len(core_indices), 1)
        nn.init.xavier_uniform_(self.core_lr.weight, gain=0.5)
        nn.init.zeros_(self.core_lr.bias)

        self.tabr = TabRSiamese(
            num_features, core_indices, hidden_dim,
            core_weight, tabr_pool, tabr_k, tabr_cos
        )

        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))

        if self.use_nam:
            self.nam = NAM(num_features, nam_hidden, nam_dropout, nam_feature_dropout)
            self.nam_gate_raw = nn.Parameter(torch.tensor(nam_gate_init))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    @property
    def nam_gate(self):
        return torch.sigmoid(self.nam_gate_raw)

    def forward(self, x, candidate_x=None, candidate_y=None, is_train=True):
        lr_logit = self.lr(x)
        core_logit = self.core_lr(x[:, self.core_indices])
        base_logit = lr_logit + core_logit

        tabr_bonus = self.tabr(x, candidate_x, candidate_y, is_train=is_train)
        out = base_logit + self.alpha * tabr_bonus
        if self.use_nam:
            out = out + self.nam_gate * self.nam(x, is_train=is_train)
        return out


class LRTabREnhanced(nn.Module):
    """Linear backbone + core linear + core-MLP interaction + TabR Siamese (+ NAM).
    """

    def __init__(self, num_features, core_indices, hidden_dim=32, dropout=0.12,
                 core_weight=8.0, tabr_pool=200, tabr_k=6, tabr_cos=True,
                 alpha_init=0.02, int_hidden=16, int_dropout=0.1, beta_init=0.1,
                 use_nam=False, nam_hidden=48, nam_dropout=0.35,
                 nam_feature_dropout=0.1, nam_gate_init=-4.0):
        super().__init__()
        self.num_features = num_features
        self.core_indices = core_indices
        self.use_nam = use_nam
        n_core = len(core_indices)

        self.lr = nn.Linear(num_features, 1)
        nn.init.xavier_uniform_(self.lr.weight, gain=0.5)
        nn.init.zeros_(self.lr.bias)

        self.core_lr = nn.Linear(n_core, 1)
        nn.init.xavier_uniform_(self.core_lr.weight, gain=0.5)
        nn.init.zeros_(self.core_lr.bias)

        self.core_int = nn.Sequential(
            nn.Linear(n_core, int_hidden),
            nn.BatchNorm1d(int_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(int_dropout),
            nn.Linear(int_hidden, 1),
        )
        nn.init.xavier_uniform_(self.core_int[-1].weight, gain=0.3)
        nn.init.zeros_(self.core_int[-1].bias)

        self.tabr = TabRSiamese(
            num_features, core_indices, hidden_dim,
            core_weight, tabr_pool, tabr_k, tabr_cos
        )

        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))
        self.beta_raw = nn.Parameter(torch.tensor(beta_init))

        if self.use_nam:
            self.nam = NAM(num_features, nam_hidden, nam_dropout, nam_feature_dropout)
            self.nam_gate_raw = nn.Parameter(torch.tensor(nam_gate_init))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    @property
    def nam_gate(self):
        return torch.sigmoid(self.nam_gate_raw)

    def forward(self, x, candidate_x=None, candidate_y=None, is_train=True):
        lr_logit = self.lr(x)
        core_logit = self.core_lr(x[:, self.core_indices])
        core_int_logit = self.core_int(x[:, self.core_indices])
        tabr_bonus = self.tabr(x, candidate_x, candidate_y, is_train=is_train)
        base_logit = lr_logit + core_logit
        out = base_logit + self.beta * core_int_logit + self.alpha * tabr_bonus
        if self.use_nam:
            out = out + self.nam_gate * self.nam(x, is_train=is_train)
        return out


class NAMTabRClassifier:
    """sklearn-style wrapper for the NAM-TabR models.
    """

    def __init__(self, model_type: str = 'enhanced', **kwargs):
        self.model_type = model_type
        self.core_indices = kwargs.get('core_feature_indices', [])
        self.hidden_dim = kwargs.get('hidden_dim', 32)
        self.dropout = kwargs.get('dropout', 0.12)
        self.lr_rate = kwargs.get('learning_rate', 9e-4)
        self.wd = kwargs.get('weight_decay', 1.2e-2)
        self.epochs = kwargs.get('num_epochs', 250)
        self.bs = kwargs.get('batch_size', 48)
        self.patience = kwargs.get('patience', 50)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.pw_mult = kwargs.get('pos_weight_multiplier', 1.1)
        self.focal_alpha = kwargs.get('focal_alpha', 0.25)
        self.focal_gamma = kwargs.get('focal_gamma', 1.8)
        self.core_weight = kwargs.get('core_weight', 8.0)
        self.tabr_pool = kwargs.get('tabr_pool_size', 200)
        self.tabr_k = kwargs.get('tabr_top_k', 6)
        self.tabr_cos = kwargs.get('tabr_cosine_sim', True)
        self.alpha_init = kwargs.get('alpha_init', 0.02)
        self.beta_init = kwargs.get('beta_init', 0.1)
        self.int_hidden = kwargs.get('int_hidden', 16)
        self.int_dropout = kwargs.get('int_dropout', 0.1)
        self.lr_warmup_epochs = kwargs.get('lr_warmup_epochs', 15)
        self.brier_lambda = kwargs.get('brier_lambda', 0.008)
        self.mixup_alpha = kwargs.get('mixup_alpha', 0.04)
        self.label_smooth = kwargs.get('label_smoothing', 0.004)
        # NAM (additive) module, gated near zero so it minimally perturbs the
        # linear baseline; frozen during the linear-warmup phase.
        self.use_nam = kwargs.get('use_nam', True)
        self.nam_hidden = kwargs.get('nam_hidden', 48)
        self.nam_dropout = kwargs.get('nam_dropout', 0.35)
        self.nam_feature_dropout = kwargs.get('nam_feature_dropout', 0.1)
        self.nam_gate_init = kwargs.get('nam_gate_init', -4.0)
        # Blend with the logistic-regression baseline (0 disables blending).
        self.blend_alpha = kwargs.get('blend_alpha', 0.15)

        self.model: Optional[nn.Module] = None
        self.lr_model: Optional[LogisticRegression] = None
        self.scaler = kwargs.get('scaler', None)
        self.feature_names: List[str] = kwargs.get('feature_names', [])
        self.train_X = None
        self.train_y = None
        self.val_auc = 0.0
        self.best_threshold = 0.5

    # ------------------------------------------------------------------ model

    def _build_model(self, num_features: int) -> nn.Module:
        nam_kw = dict(
            use_nam=self.use_nam, nam_hidden=self.nam_hidden,
            nam_dropout=self.nam_dropout, nam_feature_dropout=self.nam_feature_dropout,
            nam_gate_init=self.nam_gate_init,
        )
        if self.model_type == 'enhanced':
            return LRTabREnhanced(
                num_features=num_features, core_indices=self.core_indices,
                hidden_dim=self.hidden_dim, dropout=self.dropout,
                core_weight=self.core_weight, tabr_pool=self.tabr_pool,
                tabr_k=self.tabr_k, tabr_cos=self.tabr_cos,
                alpha_init=self.alpha_init, beta_init=self.beta_init,
                int_hidden=self.int_hidden, int_dropout=self.int_dropout,
                **nam_kw,
            )
        return LRTabRModel(
            num_features=num_features, core_indices=self.core_indices,
            hidden_dim=self.hidden_dim, dropout=self.dropout,
            core_weight=self.core_weight, tabr_pool=self.tabr_pool,
            tabr_k=self.tabr_k, tabr_cos=self.tabr_cos,
            alpha_init=self.alpha_init,
            **nam_kw,
        )

    # ------------------------------------------------------------------- fit

    def fit(self, X, y, X_val=None, y_val=None):
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1).to(self.device)
        self.train_X = Xt
        self.train_y = yt

        self.model = self._build_model(X.shape[1]).to(self.device)

        pos = y.sum()
        neg = len(y) - pos
        pw = torch.tensor([(neg / pos) * self.pw_mult]).to(self.device)
        loss_fn = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma,
                            pos_weight=pw, label_smoothing=self.label_smooth)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr_rate,
                                weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs,
                                                         eta_min=1e-6)

        yi = y.astype(int)
        sw = 1.0 / np.bincount(yi)
        sampler = WeightedRandomSampler(weights=sw[yi], num_samples=len(y),
                                        replacement=True)

        best_auc = 0.0
        best_state = None
        early_stop = 0
        has_val = X_val is not None and y_val is not None

        for epoch in range(self.epochs):
            self.model.train()
            ds = TensorDataset(Xt, yt)
            dl = DataLoader(ds, batch_size=self.bs, sampler=sampler, drop_last=False)

            # Linear-warmup: freeze all nonlinear modules and their gates.
            freeze = epoch < self.lr_warmup_epochs
            for p in self.model.tabr.parameters():
                p.requires_grad = not freeze
            self.model.alpha_raw.requires_grad = not freeze
            if self.model_type == 'enhanced':
                for p in self.model.core_int.parameters():
                    p.requires_grad = not freeze
                self.model.beta_raw.requires_grad = not freeze
            if getattr(self.model, 'use_nam', False):
                for p in self.model.nam.parameters():
                    p.requires_grad = not freeze
                self.model.nam_gate_raw.requires_grad = not freeze

            for xb, yb in dl:
                if self.mixup_alpha > 0 and not freeze and xb.size(0) > 1:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    idx = torch.randperm(xb.size(0), device=xb.device)
                    xb = lam * xb + (1 - lam) * xb[idx]
                    yb = lam * yb + (1 - lam) * yb[idx]

                logits = self.model(xb, self.train_X, self.train_y, is_train=True)
                loss = loss_fn(logits, yb)
                if self.brier_lambda > 0:
                    probs = torch.sigmoid(logits)
                    loss = loss + self.brier_lambda * F.mse_loss(probs, yb)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()

            sch.step()

            if has_val and (epoch + 1) % 5 == 0:
                val_auc = self._eval_net_auc(X_val, y_val)
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = copy.deepcopy(self.model.state_dict())
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= self.patience:
                    logger.info(f"Early stop at epoch {epoch + 1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        self.val_auc = best_auc if has_val else 0.0

        # Logistic-regression baseline for the deployed blend.
        if self.blend_alpha > 0:
            self.lr_model = LogisticRegression(max_iter=2000, C=1.0,
                                               class_weight='balanced')
            self.lr_model.fit(X, y)

        # Youden-J threshold on the validation set (blended probabilities).
        if has_val:
            self.optimize_threshold(X_val, y_val)

    # --------------------------------------------------------------- predict

    def _predict_net(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
            logits = self.model(Xt, self.train_X, self.train_y, is_train=False)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.hstack([1 - probs, probs])

    def predict_proba(self, X) -> np.ndarray:
        """Blended NAM-TabR + LR-baseline probabilities, [n_samples, 2]."""
        net_probs = self._predict_net(X)[:, 1]
        if self.lr_model is not None and self.blend_alpha > 0:
            lr_probs = self.lr_model.predict_proba(X)[:, 1]
            net_probs = self.blend_alpha * lr_probs + (1 - self.blend_alpha) * net_probs
        return np.hstack([1 - net_probs.reshape(-1, 1), net_probs.reshape(-1, 1)])

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        if threshold is None:
            threshold = self.best_threshold
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def _eval_net_auc(self, X, y) -> float:
        return roc_auc_score(y, self._predict_net(X)[:, 1])

    def evaluate_auc(self, X, y) -> float:
        """AUC of the deployed (blended) predictor."""
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    def optimize_threshold(self, X, y):
        """Find the optimal threshold with Youden's J statistic."""
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, probs)
        best_idx = np.argmax(tpr - fpr)
        self.best_threshold = float(thresholds[best_idx])
        logger.info(f"Optimal threshold: {self.best_threshold:.4f} "
                    f"(Youden J: {tpr[best_idx] - fpr[best_idx]:.4f})")

    # ------------------------------------------------------- interpretability

    def get_feature_contributions(self, X) -> List[np.ndarray]:
        """Per-feature gated NAM contributions.

        Args:
            X: scaled feature data, numpy [n_samples, n_features] or tensor.
        Returns:
            List of n_features arrays, each [n_samples]: the contribution of
            feature i to the final logit (gamma * f_NAM_i(x)).
        """
        if not getattr(self.model, 'use_nam', False):
            n = len(X)
            return [np.zeros(n) for _ in range(self.model.num_features)]

        self.model.eval()
        if not torch.is_tensor(X):
            X = torch.tensor(np.asarray(X), dtype=torch.float32)
        X = X.to(self.device)

        with torch.no_grad():
            gate = self.model.nam_gate.item()
            contributions = []
            for i, net in enumerate(self.model.nam.feature_networks):
                out = net(X[:, i:i + 1]) * gate
                contributions.append(out.detach().cpu().numpy().flatten())
        return contributions

    def get_feature_importance(self, X) -> np.ndarray:
        """Feature importance as the std of per-feature contributions."""
        return np.array([np.std(c) for c in self.get_feature_contributions(X)])

    # ------------------------------------------------------------ persistence

    def _config(self) -> Dict:
        return dict(
            model_type=self.model_type,
            core_feature_indices=list(self.core_indices),
            hidden_dim=self.hidden_dim, dropout=self.dropout,
            learning_rate=self.lr_rate, weight_decay=self.wd,
            num_epochs=self.epochs, batch_size=self.bs, patience=self.patience,
            grad_clip=self.grad_clip, pos_weight_multiplier=self.pw_mult,
            focal_alpha=self.focal_alpha, focal_gamma=self.focal_gamma,
            core_weight=self.core_weight, tabr_pool_size=self.tabr_pool,
            tabr_top_k=self.tabr_k, tabr_cosine_sim=self.tabr_cos,
            alpha_init=self.alpha_init, beta_init=self.beta_init,
            int_hidden=self.int_hidden, int_dropout=self.int_dropout,
            lr_warmup_epochs=self.lr_warmup_epochs,
            brier_lambda=self.brier_lambda, mixup_alpha=self.mixup_alpha,
            label_smoothing=self.label_smooth,
            use_nam=self.use_nam, nam_hidden=self.nam_hidden,
            nam_dropout=self.nam_dropout,
            nam_feature_dropout=self.nam_feature_dropout,
            nam_gate_init=self.nam_gate_init,
            blend_alpha=self.blend_alpha,
            feature_names=list(self.feature_names),
        )

    def save(self, path: str):
        """Save the full predictor bundle (network + LR baseline + scaler)."""
        bundle = {
            'config': self._config(),
            'state_dict': self.model.state_dict() if self.model is not None else None,
            'lr_model': self.lr_model,
            'scaler': self.scaler,
            'best_threshold': self.best_threshold,
            'val_auc': self.val_auc,
            'train_X': self.train_X.cpu() if self.train_X is not None else None,
            'train_y': self.train_y.cpu() if self.train_y is not None else None,
        }
        torch.save(bundle, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'NAMTabRClassifier':
        """Load a predictor bundle saved with :meth:`save`."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bundle = torch.load(path, map_location='cpu', weights_only=False)

        config = bundle['config']
        config['device'] = device
        clf = cls(**config)
        clf.scaler = bundle['scaler']
        clf.lr_model = bundle['lr_model']
        clf.best_threshold = bundle['best_threshold']
        clf.val_auc = bundle.get('val_auc', 0.0)

        train_X = bundle['train_X']
        clf.train_X = train_X.to(device) if train_X is not None else None
        train_y = bundle['train_y']
        clf.train_y = train_y.to(device) if train_y is not None else None

        num_features = bundle['state_dict']['lr.weight'].shape[1]
        clf.model = clf._build_model(num_features).to(device)
        clf.model.load_state_dict(bundle['state_dict'])
        clf.model.eval()
        return clf
