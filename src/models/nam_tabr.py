"""NAM-TabR Hybrid Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
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
    def __init__(self, num_features, hidden_dim=48, dropout=0.35, feature_dropout=0.1):
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

    def forward(self, x, is_train=True):
        if is_train and self.feature_dropout > 0:
            mask = torch.bernoulli(
                torch.full((x.size(0), self.num_features), 1 - self.feature_dropout, device=x.device)
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
    positive and negative neighbours.
    """
    def __init__(self, num_features, core_indices, hidden_dim=32,
                 core_weight=8.0, pool_size=300, top_k=10, use_cosine=True):
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
            nn.Linear(hidden_dim, 1)
        )
        nn.init.constant_(self.retrieval_net[-1].weight, 0.01)
        nn.init.constant_(self.retrieval_net[-1].bias, 0.0)

    def update_pool(self, x, y):
        pos_mask = (y == 1).squeeze()
        neg_mask = (y == 0).squeeze()
        pos_x = x[pos_mask]
        neg_x = x[neg_mask]

        if pos_x.size(0) > 0:
            n_new = min(pos_x.size(0), self.pool_size)
            start = self.pos_ptr.item() % self.pool_size
            for i in range(n_new):
                idx = (start + i) % self.pool_size
                self.pos_pool[idx] = pos_x[i]
            self.pos_ptr += n_new
            if not self.pos_filled and self.pos_ptr >= self.pool_size:
                self.pos_filled.fill_(True)

        if neg_x.size(0) > 0:
            n_new = min(neg_x.size(0), self.pool_size)
            start = self.neg_ptr.item() % self.pool_size
            for i in range(n_new):
                idx = (start + i) % self.pool_size
                self.neg_pool[idx] = neg_x[i]
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
    """LR backbone + core-feature linear term + TabR Siamese retrieval.

    Optionally adds a gated NAM (additive) term. The NAM gate is initialised
    near zero (``nam_gate_init`` very negative) so the model starts equivalent
    to the LR+TabR baseline and only grows the NAM contribution if it helps.
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
    """LR backbone + core-feature linear + core MLP interaction + TabR Siamese.

    Optionally adds a gated NAM (additive) term, initialised near zero so it
    only contributes if it reduces the loss.
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
