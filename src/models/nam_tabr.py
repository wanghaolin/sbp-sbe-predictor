"""NAM-TabR Hybrid Model"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class NAM(nn.Module):
    """Neural Additive Model Module"""
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 48,
        dropout: float = 0.35,
        zero_mean: bool = True,
        feature_dropout: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.zero_mean = zero_mean
        self.feature_dropout = feature_dropout
        
        self.feature_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SELU(),
                nn.AlphaDropout(dropout),
                nn.Linear(hidden_dim // 2, 1, bias=False)
            )
            for _ in range(num_features)
        ])
        
        self.register_buffer('feature_mask', torch.ones(num_features))
        
        for net in self.feature_networks:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_features]
            is_train: whether in training mode
        Returns:
            nam_output: [batch_size, 1]
        """
        if is_train and self.feature_dropout > 0:
            mask = torch.bernoulli(
                torch.full((x.size(0), self.num_features), 1 - self.feature_dropout, device=x.device)
            )
            x = x * mask
        
        nam_components = []
        for i, net in enumerate(self.feature_networks):
            feature = x[:, i:i+1]
            out = net(feature)
            if self.zero_mean:
                out = out - out.mean(dim=0, keepdim=True)
            nam_components.append(out)
        
        nam_output = torch.cat(nam_components, dim=1).sum(dim=1, keepdim=True)
        return nam_output


class TabR(nn.Module):
    """Table Retrieval Module"""
    def __init__(
        self,
        num_features: int,
        core_indices: List[int],
        hidden_dim: int = 48,
        dropout: float = 0.35,
        core_weight: float = 5.0,
        pool_size: int = 100,
        top_k: int = 3,
        use_cosine: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.core_indices = core_indices
        self.pool_size = pool_size
        self.top_k = top_k
        self.use_cosine = use_cosine
        
        self.register_buffer('feature_weights', torch.ones(num_features))
        for idx in core_indices:
            self.feature_weights[idx] = core_weight
        
        self.register_buffer('anchor_pool', torch.zeros(pool_size, num_features))
        self.register_buffer('pool_counter', torch.zeros(1, dtype=torch.long))
        self.register_buffer('pool_filled', torch.zeros(1, dtype=torch.bool))
        
        self.retrieval_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1)
        )
        
        nn.init.constant_(self.retrieval_proj[-1].weight, 0.01)
        nn.init.constant_(self.retrieval_proj[-1].bias, -2.0)
    
    def update_pool(self, x: torch.Tensor, y: torch.Tensor):
        """Update anchor pool"""
        pos_mask = (y == 1).squeeze()
        pos_x = x[pos_mask]
        
        if pos_x.size(0) > 0:
            n_new = min(pos_x.size(0), self.pool_size)
            start_idx = self.pool_counter.item() % self.pool_size
            
            for i in range(n_new):
                idx = (start_idx + i) % self.pool_size
                self.anchor_pool[idx] = pos_x[i]
            
            self.pool_counter += n_new
            if not self.pool_filled and self.pool_counter >= self.pool_size:
                self.pool_filled.fill_(True)
    
    def forward(
        self,
        x: torch.Tensor,
        candidate_x: Optional[torch.Tensor] = None,
        candidate_y: Optional[torch.Tensor] = None,
        is_train: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_features]
            candidate_x: [n_candidates, num_features]
            candidate_y: [n_candidates, 1]
            is_train: whether in training mode
        Returns:
            retrieval_bonus: [batch_size, 1]
        """
        retrieval_bonus = torch.zeros(x.size(0), 1, device=x.device)
        
        if is_train and candidate_x is not None and candidate_y is not None:
            self.update_pool(candidate_x, candidate_y)
        
        if self.pool_filled or self.pool_counter > 0:
            pool_size = min(self.pool_counter.item(), self.pool_size)
            anchors = self.anchor_pool[:pool_size]
            
            weighted_x = x * self.feature_weights
            weighted_anchors = anchors * self.feature_weights
            
            if self.use_cosine:
                x_norm = F.normalize(weighted_x, dim=-1)
                anchor_norm = F.normalize(weighted_anchors, dim=-1)
                sim = x_norm @ anchor_norm.T
            else:
                dist = torch.cdist(weighted_x, weighted_anchors)
                sim = torch.exp(-dist.pow(2) / (2 * self.num_features))
            
            top_k = min(self.top_k, sim.size(-1))
            top_sim = sim.topk(top_k, dim=-1)[0]
            avg_sim = top_sim.mean(dim=-1, keepdim=True)
            
            retrieval_bonus = self.retrieval_proj(avg_sim)
        
        return retrieval_bonus


class UncertaintyGate(nn.Module):
    """Uncertainty-based Gating Mechanism"""
    def __init__(self, initial_bias: float = -4.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].weight, 2.0)
        nn.init.constant_(self.gate[0].bias, initial_bias)
    
    def forward(self, lr_logit: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lr_logit: [batch_size, 1]
        Returns:
            gate: [batch_size, 1]
        """
        uncertainty = torch.abs(lr_logit)
        gate_input = -uncertainty
        gate = self.gate(gate_input)
        return gate


class LNTHModel(nn.Module):
    """LNTH Model - Linear + NAM + TabR Hybrid"""
    def __init__(
        self,
        num_features: int,
        core_indices: List[int],
        hidden_dim: int = 48,
        dropout: float = 0.35,
        core_weight: float = 5.0,
        linear_warmup_epochs: int = 15,
        gate_initial_bias: float = -4.0,
        feature_dropout: float = 0.1,
        tabr_pool_size: int = 100,
        tabr_top_k: int = 3,
        tabr_use_cosine: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.core_indices = core_indices
        self.linear_warmup_epochs = linear_warmup_epochs
        self.current_epoch = 0
        
        self.feature_bn = nn.BatchNorm1d(num_features)
        
        self.linear_backbone = nn.Linear(num_features, 1)
        nn.init.xavier_uniform_(self.linear_backbone.weight, gain=0.5)
        nn.init.zeros_(self.linear_backbone.bias)
        
        self.core_linear = nn.Linear(len(core_indices), 1)
        nn.init.xavier_uniform_(self.core_linear.weight, gain=0.5)
        nn.init.zeros_(self.core_linear.bias)
        
        self.nam = NAM(
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            zero_mean=True,
            feature_dropout=feature_dropout
        )
        
        self.tabr = TabR(
            num_features=num_features,
            core_indices=core_indices,
            hidden_dim=hidden_dim,
            dropout=dropout,
            core_weight=core_weight,
            pool_size=tabr_pool_size,
            top_k=tabr_top_k,
            use_cosine=tabr_use_cosine
        )
        
        self.core_interaction = nn.Sequential(
            nn.Linear(len(core_indices), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.uncertainty_gate = UncertaintyGate(initial_bias=gate_initial_bias)
    
    def set_epoch(self, epoch: int):
        """Set current epoch"""
        self.current_epoch = epoch
    
    def forward(
        self,
        x: torch.Tensor,
        candidate_x: Optional[torch.Tensor] = None,
        candidate_y: Optional[torch.Tensor] = None,
        is_train: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_features]
            candidate_x: [n_candidates, num_features]
            candidate_y: [n_candidates, 1]
            is_train: whether in training mode
        Returns:
            final_logit: [batch_size, 1]
        """
        x_normalized = self.feature_bn(x)
        
        linear_logit = self.linear_backbone(x_normalized)
        
        core_features = x_normalized[:, self.core_indices]
        core_logit = self.core_linear(core_features)
        
        lr_logit = linear_logit + core_logit
        
        nam_logit = self.nam(x_normalized, is_train=is_train)
        
        tabr_logit = self.tabr(x_normalized, candidate_x, candidate_y, is_train=is_train)
        
        interaction_logit = self.core_interaction(core_features)
        
        gate = self.uncertainty_gate(lr_logit)
        
        if self.current_epoch < self.linear_warmup_epochs:
            gate = torch.zeros_like(gate)
        
        nonlinear_logit = nam_logit + tabr_logit + interaction_logit
        final_logit = lr_logit + gate * nonlinear_logit
        
        return final_logit


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, 1]
            targets: [batch_size, 1]
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        if self.pos_weight is not None:
            pos_weight = targets * (self.pos_weight - 1) + 1
            alpha_weight = alpha_weight * pos_weight
        
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


class LNTHClassifier:
    """LNTH Classifier"""
    def __init__(self, **kwargs):
        self.hidden_dim = kwargs.get('hidden_dim', 48)
        self.dropout = kwargs.get('dropout', 0.35)
        
        self.core_feature_indices = kwargs.get('core_feature_indices', [])
        self.core_weight = kwargs.get('core_weight', 5.0)
        
        self.linear_warmup_epochs = kwargs.get('linear_warmup_epochs', 15)
        self.gate_initial_bias = kwargs.get('gate_initial_bias', -4.0)
        
        self.feature_dropout = kwargs.get('feature_dropout', 0.1)
        
        self.tabr_pool_size = kwargs.get('tabr_pool_size', 100)
        self.tabr_top_k = kwargs.get('tabr_top_k', 3)
        self.tabr_use_cosine = kwargs.get('tabr_use_cosine', True)
        
        self.learning_rate = kwargs.get('learning_rate', 8e-5)
        self.weight_decay = kwargs.get('weight_decay', 2e-2)
        self.num_epochs = kwargs.get('num_epochs', 400)
        self.batch_size = kwargs.get('batch_size', 64)
        self.patience = kwargs.get('patience', 80)
        self.grad_clip = kwargs.get('grad_clip', 0.5)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pos_weight_multiplier = kwargs.get('pos_weight_multiplier', 1.0)
        self.use_focal_loss = kwargs.get('use_focal_loss', True)
        self.focal_alpha = kwargs.get('focal_alpha', 0.35)
        self.focal_gamma = kwargs.get('focal_gamma', 2.0)
        
        self.model = None
        self.train_X_tensor = None
        self.train_y_tensor = None
        self.best_threshold = 0.5
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model"""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.train_X_tensor = X_tensor
        self.train_y_tensor = y_tensor
        
        self.model = LNTHModel(
            num_features=X.shape[1],
            core_indices=self.core_feature_indices,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            core_weight=self.core_weight,
            linear_warmup_epochs=self.linear_warmup_epochs,
            gate_initial_bias=self.gate_initial_bias,
            feature_dropout=self.feature_dropout,
            tabr_pool_size=self.tabr_pool_size,
            tabr_top_k=self.tabr_top_k,
            tabr_use_cosine=self.tabr_use_cosine
        ).to(self.device)
        
        pos_num = y.sum()
        neg_num = len(y) - pos_num
        calculated_weight = (neg_num / pos_num) * self.pos_weight_multiplier
        pos_weight = torch.tensor([calculated_weight]).to(self.device)
        
        if self.use_focal_loss:
            loss_fn = FocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                pos_weight=pos_weight
            )
        else:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=80, T_mult=2, eta_min=1e-6
        )
        
        y_int = y.astype(int)
        class_counts = np.bincount(y_int)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_int]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y),
            replacement=True
        )
        
        best_auc = 0.0
        best_state = None
        early_stop_count = 0
        
        for epoch in range(self.num_epochs):
            self.model.set_epoch(epoch)
            self.model.train()
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler
            )
            
            epoch_loss = 0.0
            for xb, yb in dataloader:
                logits = self.model(
                    xb,
                    self.train_X_tensor,
                    self.train_y_tensor,
                    is_train=True
                )
                loss = loss_fn(logits, yb)
                
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip
                )
                
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if X_val is not None:
                current_auc = self.evaluate_auc(X_val, y_val)
                
                if (epoch + 1) % 40 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs}, "
                        f"Val AUC: {current_auc:.4f}, "
                        f"Loss: {epoch_loss/len(dataloader):.4f}, "
                        f"LR: {current_lr:.6f}"
                    )
                
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_state = copy.deepcopy(self.model.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                
                if early_stop_count >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        if best_state:
            self.model.load_state_dict(best_state)
            logger.info(f"Best validation AUC: {best_auc:.4f}")
        
        # Optimize threshold on validation set
        if X_val is not None:
            self.optimize_threshold(X_val, y_val)
    
    def optimize_threshold(self, X, y):
        """Find optimal threshold using Youden's J statistic"""
        from sklearn.metrics import roc_curve
        
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, probs)
        
        # Youden's J statistic: maximize sensitivity + specificity - 1
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        self.best_threshold = thresholds[best_idx]
        
        logger.info(f"Optimal threshold: {self.best_threshold:.4f} (Youden J: {youden_j[best_idx]:.4f})")
    
    def evaluate_auc(self, X, y):
        """Evaluate AUC"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(
                X_t,
                self.train_X_tensor,
                self.train_y_tensor,
                is_train=False
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            return roc_auc_score(y, probs)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(
                X_t,
                self.train_X_tensor,
                self.train_y_tensor,
                is_train=False
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            return np.hstack([1 - probs, probs])
    
    def predict(self, X, threshold=None):
        """Predict class labels"""
        if threshold is None:
            threshold = self.best_threshold
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
    
    def get_feature_importance(self, X):
        """
        Calculate feature importance based on contribution standard deviation
        
        Args:
            X: feature data [n_samples, n_features]
        Returns:
            feature importance array [n_features]
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        feature_contributions = self.get_feature_contributions(X_tensor)
        
        feature_importance = []
        for i, contribution in enumerate(feature_contributions):
            importance = np.std(contribution)
            feature_importance.append(importance)
        
        return np.array(feature_importance)
    
    def get_feature_contributions(self, X):
        """
        Get feature contributions
        
        Args:
            X: feature data [n_samples, n_features]
        Returns:
            feature contribution list [n_features], each element is [n_samples] array
        """
        self.model.eval()
        
        with torch.no_grad():
            X_normalized = self.model.feature_bn(X)
            
            nam_module = self.model.nam
            
            feature_contributions = []
            for i, net in enumerate(nam_module.feature_networks):
                feature = X_normalized[:, i:i+1]
                out = net(feature)
                if nam_module.zero_mean:
                    pass
                feature_contributions.append(out.detach().cpu().numpy().flatten())
            
            return feature_contributions
