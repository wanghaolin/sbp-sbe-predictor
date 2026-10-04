#!/usr/bin/env python3
"""SBP (ascites) NAM-TabR model training script.

Trains the NAM-TabR hybrid (LRTabREnhanced: linear backbone + core-feature
interaction + Siamese TabR retrieval + gated NAM).
"""

import os
import sys
import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data.data_loader import load_data
from models.nam_tabr import NAMTabRClassifier

# ------------------------------------------------------------------- config
FLUID = 'sbp'
LABEL_COL = ''
FEATURES = []
CORE_FEATURES = []

MODEL_CONFIG = dict(
    model_type='enhanced',
    hidden_dim=32,
    dropout=0.10,
    learning_rate=1e-3,
    weight_decay=1e-2,
    num_epochs=250,
    patience=50,
    batch_size=48,
    core_weight=8.0,
    focal_alpha=0.25,
    focal_gamma=1.8,
    pos_weight_multiplier=1.1,
    tabr_pool_size=200,
    tabr_top_k=6,
    tabr_cosine_sim=True,
    lr_warmup_epochs=15,
    brier_lambda=0.008,
    mixup_alpha=0.04,
    label_smoothing=0.004,
    alpha_init=0.02,
    beta_init=0.1,
    int_hidden=16,
    int_dropout=0.1,
    use_nam=True,
    blend_alpha=0.15,
)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred_proba, threshold) -> Dict[str, float]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def save_explanations(model, X_raw, X_scaled, y, probs, preds, path):
    """Per-sample feature contributions (gated NAM) + raw feature values."""
    contributions = model.get_feature_contributions(X_scaled)
    rows = []
    for i in range(len(y)):
        row = {
            'sample_index': i,
            'y_true': int(y[i]),
            'y_pred_proba': float(probs[i]),
            'y_pred': int(preds[i]),
        }
        for j, name in enumerate(FEATURES):
            row[name] = float(X_raw[i, j])
            row[f'{name}_contribution'] = float(contributions[j][i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Sample explanations saved to: {path}")


def main():
    # Analysis artefacts (metrics, predictions, per-sample explanations) go to
    # outputs/<fluid>/; the deployable predictor bundle goes to
    # checkpoints/<fluid>/model.pt, where the Gradio app loads it from.
    output_dir = os.path.join(project_dir, 'outputs', FLUID)
    metrics_dir = os.path.join(output_dir, 'metrics')
    predictions_dir = os.path.join(output_dir, 'predictions')
    interpretability_dir = os.path.join(output_dir, 'interpretability')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints', FLUID)
    for d in (metrics_dir, predictions_dir, interpretability_dir, checkpoints_dir):
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"NAM-TabR Training - {FLUID.upper()}")
    logger.info("=" * 80)

    core_feature_indices = [FEATURES.index(f) for f in CORE_FEATURES]
    logger.info(f"Features: {FEATURES}")
    logger.info(f"Core features: {CORE_FEATURES} (indices {core_feature_indices})")

    data = load_data(os.path.join(project_dir, 'data', FLUID))
    splits = {}
    for name in ('train', 'internal_test', 'external_test'):
        df = data[name]
        splits[name] = (
            df[FEATURES].values.astype(np.float32),
            df[LABEL_COL].values.astype(np.float32),
        )
        logger.info(f"{name}: {df.shape}, positives={int(splits[name][1].sum())}")

    (X_train, y_train) = splits['train']
    (X_internal, y_internal) = splits['internal_test']
    (X_external, y_external) = splits['external_test']

    # Median imputation (fitted on train only) + standardisation.
    medians = np.nanmedian(X_train, axis=0)
    def impute(X):
        X = X.copy()
        idx = np.where(np.isnan(X))
        X[idx] = np.take(medians, idx[1])
        return X

    X_train, X_internal, X_external = impute(X_train), impute(X_internal), impute(X_external)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_internal_scaled = scaler.transform(X_internal)
    X_external_scaled = scaler.transform(X_external)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    set_seed(42)

    model = NAMTabRClassifier(
        core_feature_indices=core_feature_indices,
        device=device,
        feature_names=FEATURES,
        scaler=scaler,
        **MODEL_CONFIG,
    )

    logger.info("Training NAM-TabR (linear warmup -> joint training)...")
    model.fit(X_train_scaled, y_train, X_internal_scaled, y_internal)

    internal_probs = model.predict_proba(X_internal_scaled)[:, 1]
    external_probs = model.predict_proba(X_external_scaled)[:, 1]
    threshold = model.best_threshold
    internal_preds = (internal_probs >= threshold).astype(int)
    external_preds = (external_probs >= threshold).astype(int)

    internal_metrics = calculate_metrics(y_internal, internal_probs, threshold)
    external_metrics = calculate_metrics(y_external, external_probs, threshold)

    logger.info("-" * 80)
    logger.info(f"Optimal threshold (Youden J): {threshold:.4f}")
    logger.info(f"Internal: AUC={internal_metrics['auc']:.4f} "
                f"Acc={internal_metrics['accuracy']:.4f} "
                f"Prec={internal_metrics['precision']:.4f} "
                f"Rec={internal_metrics['recall']:.4f} F1={internal_metrics['f1']:.4f}")
    logger.info(f"External: AUC={external_metrics['auc']:.4f} "
                f"Acc={external_metrics['accuracy']:.4f} "
                f"Prec={external_metrics['precision']:.4f} "
                f"Rec={external_metrics['recall']:.4f} F1={external_metrics['f1']:.4f}")

    # Feature importance (std of gated NAM contributions on the external set).
    importance = model.get_feature_importance(X_external_scaled)
    importance_df = pd.DataFrame({
        'feature': FEATURES, 'importance': importance,
    }).sort_values('importance', ascending=False)
    importance_path = os.path.join(metrics_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info("Feature importance: " +
                ", ".join(f"{r['feature']}={r['importance']:.4f}"
                          for _, r in importance_df.iterrows()))

    # Interpretability outputs (raw feature values + per-feature contributions).
    save_explanations(model, X_train, X_train_scaled, y_train,
                      model.predict_proba(X_train_scaled)[:, 1],
                      (model.predict_proba(X_train_scaled)[:, 1] >= threshold).astype(int),
                      os.path.join(interpretability_dir, 'train_sample_explanations.csv'))
    save_explanations(model, X_internal, X_internal_scaled, y_internal,
                      internal_probs, internal_preds,
                      os.path.join(interpretability_dir, 'internal_sample_explanations.csv'))
    save_explanations(model, X_external, X_external_scaled, y_external,
                      external_probs, external_preds,
                      os.path.join(interpretability_dir, 'external_sample_explanations.csv'))

    # Predictions.
    for name, y, probs, preds in (('internal', y_internal, internal_probs, internal_preds),
                                  ('external', y_external, external_probs, external_preds)):
        path = os.path.join(predictions_dir, f'{name}_predictions.csv')
        pd.DataFrame({'y_true': y, 'y_pred_proba': probs, 'y_pred': preds}).to_csv(path, index=False)

    # Full predictor bundle (network + LR baseline + scaler + threshold).
    model_path = os.path.join(checkpoints_dir, 'model.pt')
    model.save(model_path)
    logger.info(f"Model bundle saved to: {model_path}")

    results = {
        'fluid': FLUID,
        'features': FEATURES,
        'core_features': CORE_FEATURES,
        'core_feature_indices': core_feature_indices,
        'optimal_threshold': float(threshold),
        'blend_alpha': MODEL_CONFIG['blend_alpha'],
        'val_auc': float(model.val_auc),
        'metrics': {'internal': internal_metrics, 'external': external_metrics},
        'output_files': {'model': model_path},
    }
    results_path = os.path.join(metrics_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
