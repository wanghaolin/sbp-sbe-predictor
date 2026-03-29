#!/usr/bin/env python3
"""Model training script"""

import os
import sys
import json
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

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
from models.nam_tabr import LNTHClassifier


def set_seed(seed: int = 42):
    """Set random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def main():
    """Main function"""
    
    temp_dir = os.path.join(project_dir, 'temp')
    metrics_dir = os.path.join(temp_dir, 'metrics')
    models_dir = os.path.join(temp_dir, 'models')
    predictions_dir = os.path.join(temp_dir, 'predictions')
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Model Training")
    logger.info("=" * 80)
    
    best_features = ['PMN', 'Ascitic fluid WBC', 'Total cell count', 'Lymphocyte percentage']
    logger.info(f"Features: {best_features}")
    
    core_features = ['PMN']
    core_feature_indices = [best_features.index(f) for f in core_features]
    logger.info(f"Core features: {core_features}")
    logger.info(f"Core feature indices: {core_feature_indices}")
    
    logger.info("Loading data...")
    data_dir = os.path.join(project_dir, 'data')
    data = load_data(data_dir)
    
    train_df = data['train']
    internal_test_df = data['internal_test']
    external_test_df = data['external_test']
    
    label_col = 'PMN%'
    
    logger.info(f"Train set: {train_df.shape}")
    logger.info(f"Internal test set: {internal_test_df.shape}")
    logger.info(f"External test set: {external_test_df.shape}")
    
    X_train = train_df[best_features].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.float32)
    
    X_internal = internal_test_df[best_features].values.astype(np.float32)
    y_internal = internal_test_df[label_col].values.astype(np.float32)
    
    X_external = external_test_df[best_features].values.astype(np.float32)
    y_external = external_test_df[label_col].values.astype(np.float32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_internal_scaled = scaler.transform(X_internal)
    X_external_scaled = scaler.transform(X_external)
    
    scaler_path = os.path.join(temp_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to: {scaler_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    set_seed(42)
    
    logger.info("Creating model...")
    model = LNTHClassifier(
        hidden_dim=64,
        dropout=0.35,
        core_feature_indices=core_feature_indices,
        core_weight=7.0,
        linear_warmup_epochs=20,
        gate_initial_bias=-5.0,
        feature_dropout=0.15,
        tabr_pool_size=150,
        tabr_top_k=5,
        tabr_use_cosine=True,
        learning_rate=1e-4,
        batch_size=64,
        num_epochs=400,
        patience=100,
        weight_decay=5e-2,
        grad_clip=0.5,
        pos_weight_multiplier=0.8,
        device=device
    )
    
    logger.info("Training...")
    model.fit(X_train_scaled, y_train, X_internal_scaled, y_internal)
    
    logger.info("Evaluating...")
    internal_auc = model.evaluate_auc(X_internal_scaled, y_internal)
    external_auc = model.evaluate_auc(X_external_scaled, y_external)
    
    logger.info("Computing feature importance...")
    feature_importance = model.get_feature_importance(X_external_scaled)
    
    importance_df = pd.DataFrame({
        'feature': best_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(metrics_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to: {importance_path}")
    logger.info("Global Feature Importance:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.6f}")
    
    internal_probs = model.predict_proba(X_internal_scaled)[:, 1]
    external_probs = model.predict_proba(X_external_scaled)[:, 1]
    
    # Use optimized threshold from model
    optimal_threshold = model.best_threshold
    internal_preds = (internal_probs >= optimal_threshold).astype(int)
    external_preds = (external_probs >= optimal_threshold).astype(int)
    
    logger.info(f"Using optimal threshold: {optimal_threshold:.4f}")
    
    logger.info("Computing individual sample explanations...")
    interpretability_dir = os.path.join(temp_dir, 'interpretability')
    os.makedirs(interpretability_dir, exist_ok=True)
    
    X_external_tensor = torch.tensor(X_external_scaled, dtype=torch.float32).to(device)
    feature_contributions = model.get_feature_contributions(X_external_tensor)
    
    sample_explanations = []
    for i in range(len(y_external)):
        explanation = {
            'sample_index': i,
            'y_true': int(y_external[i]),
            'y_pred_proba': float(external_probs[i]),
            'y_pred': int(external_preds[i])
        }
        for j, feat_name in enumerate(best_features):
            explanation[f'{feat_name}_contribution'] = float(feature_contributions[j][i])
            explanation[f'{feat_name}_value'] = float(X_external[i, j])
        sample_explanations.append(explanation)
    
    explanations_df = pd.DataFrame(sample_explanations)
    explanations_path = os.path.join(interpretability_dir, 'sample_explanations.csv')
    explanations_df.to_csv(explanations_path, index=False)
    logger.info(f"Sample explanations saved to: {explanations_path}")
    
    # Generate sample explanations for training set
    logger.info("Computing training set sample explanations...")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    train_feature_contributions = model.get_feature_contributions(X_train_tensor)
    
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)
    
    train_sample_explanations = []
    for i in range(len(y_train)):
        explanation = {
            'sample_index': i,
            'y_true': int(y_train[i]),
            'y_pred_proba': float(train_probs[i]),
            'y_pred': int(train_preds[i])
        }
        for j, feat_name in enumerate(best_features):
            explanation[f'{feat_name}_contribution'] = float(train_feature_contributions[j][i])
            explanation[f'{feat_name}_value'] = float(X_train[i, j])
        train_sample_explanations.append(explanation)
    
    train_explanations_df = pd.DataFrame(train_sample_explanations)
    train_explanations_path = os.path.join(interpretability_dir, 'train_sample_explanations.csv')
    train_explanations_df.to_csv(train_explanations_path, index=False)
    logger.info(f"Training set sample explanations saved to: {train_explanations_path}")
    
    # Generate sample explanations for internal test set
    logger.info("Computing internal test set sample explanations...")
    X_internal_tensor = torch.tensor(X_internal_scaled, dtype=torch.float32).to(device)
    internal_feature_contributions = model.get_feature_contributions(X_internal_tensor)
    
    internal_sample_explanations = []
    for i in range(len(y_internal)):
        explanation = {
            'sample_index': i,
            'y_true': int(y_internal[i]),
            'y_pred_proba': float(internal_probs[i]),
            'y_pred': int(internal_preds[i])
        }
        for j, feat_name in enumerate(best_features):
            explanation[f'{feat_name}_contribution'] = float(internal_feature_contributions[j][i])
            explanation[f'{feat_name}_value'] = float(X_internal[i, j])
        internal_sample_explanations.append(explanation)
    
    internal_explanations_df = pd.DataFrame(internal_sample_explanations)
    internal_explanations_path = os.path.join(interpretability_dir, 'internal_sample_explanations.csv')
    internal_explanations_df.to_csv(internal_explanations_path, index=False)
    logger.info(f"Internal test set sample explanations saved to: {internal_explanations_path}")
    
    # Rename external explanations file for clarity
    external_explanations_path = os.path.join(interpretability_dir, 'external_sample_explanations.csv')
    explanations_df.to_csv(external_explanations_path, index=False)
    logger.info(f"External test set sample explanations saved to: {external_explanations_path}")
    
    positive_samples = [i for i in range(len(y_external)) if y_external[i] == 1][:5]
    negative_samples = [i for i in range(len(y_external)) if y_external[i] == 0][:5]
    sample_indices = positive_samples + negative_samples
    
    detailed_explanations = []
    for idx in sample_indices:
        explanation = {
            'sample_index': idx,
            'y_true': int(y_external[idx]),
            'y_pred_proba': float(external_probs[idx]),
            'y_pred': int(external_preds[idx]),
            'features': {}
        }
        for j, feat_name in enumerate(best_features):
            explanation['features'][feat_name] = {
                'value': float(X_external[idx, j]),
                'contribution': float(feature_contributions[j][idx])
            }
        detailed_explanations.append(explanation)
    
    detailed_path = os.path.join(interpretability_dir, 'detailed_explanations.json')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_explanations, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed explanations saved to: {detailed_path}")
    
    contribution_summary = {
        'feature': best_features,
        'mean_contribution': [float(np.mean(c)) for c in feature_contributions],
        'std_contribution': [float(np.std(c)) for c in feature_contributions],
        'min_contribution': [float(np.min(c)) for c in feature_contributions],
        'max_contribution': [float(np.max(c)) for c in feature_contributions]
    }
    contribution_df = pd.DataFrame(contribution_summary)
    contribution_path = os.path.join(interpretability_dir, 'contribution_summary.csv')
    contribution_df.to_csv(contribution_path, index=False)
    logger.info(f"Contribution summary saved to: {contribution_path}")
    
    internal_metrics = calculate_metrics(y_internal, internal_probs, optimal_threshold)
    external_metrics = calculate_metrics(y_external, external_probs, optimal_threshold)
    
    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Features: {best_features}")
    logger.info(f"Core features: {core_features}")
    logger.info("-" * 80)
    logger.info("Internal Validation:")
    logger.info(f"  AUC: {internal_auc:.4f}")
    logger.info(f"  Accuracy: {internal_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {internal_metrics['precision']:.4f}")
    logger.info(f"  Recall: {internal_metrics['recall']:.4f}")
    logger.info(f"  F1: {internal_metrics['f1']:.4f}")
    logger.info("-" * 80)
    logger.info("External Validation:")
    logger.info(f"  AUC: {external_auc:.4f}")
    logger.info(f"  Accuracy: {external_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {external_metrics['precision']:.4f}")
    logger.info(f"  Recall: {external_metrics['recall']:.4f}")
    logger.info(f"  F1: {external_metrics['f1']:.4f}")
    logger.info("=" * 80)
    
    model_path = os.path.join(models_dir, 'model.pt')
    torch.save(model.model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    internal_predictions_df = pd.DataFrame({
        'y_true': y_internal,
        'y_pred_proba': internal_probs,
        'y_pred': internal_preds
    })
    internal_pred_path = os.path.join(predictions_dir, 'internal_predictions.csv')
    internal_predictions_df.to_csv(internal_pred_path, index=False)
    logger.info(f"Internal predictions saved to: {internal_pred_path}")
    
    external_predictions_df = pd.DataFrame({
        'y_true': y_external,
        'y_pred_proba': external_probs,
        'y_pred': external_preds
    })
    external_pred_path = os.path.join(predictions_dir, 'external_predictions.csv')
    external_predictions_df.to_csv(external_pred_path, index=False)
    logger.info(f"External predictions saved to: {external_pred_path}")
    
    results = {
        'feature_combo': best_features,
        'core_features': core_features,
        'core_feature_indices': core_feature_indices,
        'optimal_threshold': float(optimal_threshold),
        'metrics': {
            'internal': {
                'auc': float(internal_auc),
                'accuracy': float(internal_metrics['accuracy']),
                'precision': float(internal_metrics['precision']),
                'recall': float(internal_metrics['recall']),
                'f1': float(internal_metrics['f1'])
            },
            'external': {
                'auc': float(external_auc),
                'accuracy': float(external_metrics['accuracy']),
                'precision': float(external_metrics['precision']),
                'recall': float(external_metrics['recall']),
                'f1': float(external_metrics['f1'])
            }
        },
        'output_files': {
            'model': model_path,
            'scaler': scaler_path,
            'internal_predictions': internal_pred_path,
            'external_predictions': external_pred_path,
            'feature_importance': importance_path,
            'train_sample_explanations': train_explanations_path,
            'internal_sample_explanations': internal_explanations_path,
            'external_sample_explanations': external_explanations_path,
            'detailed_explanations': detailed_path,
            'contribution_summary': contribution_path
        }
    }
    
    results_path = os.path.join(metrics_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
