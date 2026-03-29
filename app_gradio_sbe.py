#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import gradio as gr
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# Path setup
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

from models.nam_tabr import LNTHClassifier, LNTHModel

# Constants
FEATURE_NAMES = ['PMN%', 'PMN', 'Ascitic WBC', 'WBC', 'CRP']

# Model parameters
SHARED_PARAMS = {
    'hidden_dim': 64, 'dropout': 0.35, 'core_weight': 7.0,
    'linear_warmup_epochs': 20, 'gate_initial_bias': -5.0,
    'feature_dropout': 0.15, 'tabr_pool_size': 150, 'tabr_top_k': 5, 'tabr_use_cosine': True
}

# Load artifacts
temp_dir = os.path.join(project_dir, 'temp')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = joblib.load(os.path.join(temp_dir, 'scaler.pkl'))

model = LNTHClassifier(core_feature_indices=[0, 1], device=device, **SHARED_PARAMS)
model.model = LNTHModel(num_features=5, core_indices=[0, 1], **SHARED_PARAMS).to(device)
model.model.load_state_dict(torch.load(os.path.join(temp_dir, 'models', 'model.pt'), map_location=device))
model.model.eval()
model.train_X_tensor = torch.zeros(1, 5, dtype=torch.float32).to(device)
model.train_y_tensor = torch.zeros(1, 1, dtype=torch.float32).to(device)

# Load sample explanations
sample_explanations_path = os.path.join(temp_dir, 'interpretability', 'external_sample_explanations.csv')
sample_df = pd.read_csv(sample_explanations_path) if os.path.exists(sample_explanations_path) else None

# Column names in explanations file
VALUE_COLS = ['PMN%', 'PMN', 'Ascitic WBC', 'WBC', 'CRP']

def get_random_sample():
    if sample_df is None or len(sample_df) == 0:
        return [20.0, 50.0, 200.0, 5.0, 20.0, "N/A"]
    
    sample = sample_df.sample(1).iloc[0]
    values = [float(round(sample[col], 1)) for col in VALUE_COLS]
    label = "Positive" if sample['y_true'] == 1 else "Negative"
    values.append(label)
    return values

def create_chart(contributions):
    """Create feature contribution bar chart"""
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    
    colors = ['#3B82F6' if c >= 0 else '#EF4444' for c in contributions]
    bars = ax.barh(FEATURE_NAMES, contributions, color=colors, height=0.65, 
                   edgecolor='white', linewidth=2, alpha=0.85)
    
    ax.axvline(0, color='#CBD5E1', linewidth=1.2)
    ax.set_xlabel('Contribution', fontsize=10, color='#64748B', labelpad=8)
    ax.tick_params(left=False, bottom=False, labelsize=10, colors='#1E293B')
    
    for bar, c in zip(bars, contributions):
        ax.text(bar.get_width() + (0.015 if c >= 0 else -0.015),
                bar.get_y() + bar.get_height()/2, f'{c:.3f}',
                ha='left' if c >= 0 else 'right', va='center',
                fontsize=9, color='#334155')
    
    ax.set_xlim(-0.4, 0.4)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='#E2E8F0')
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
    return fig

def predict(pmn_ratio, pmn, wbc_fluid, wbc, crp):
    """Generate prediction and explanation"""
    X = np.array([[pmn_ratio, pmn, wbc_fluid, wbc, crp]], dtype=np.float32)
    X_scaled = scaler.transform(X)
    
    proba = model.predict_proba(X_scaled)[0, 1]
    prediction = int(proba >= 0.5)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    contributions = [float(c[0]) for c in model.get_feature_contributions(X_tensor)]
    
    fig = create_chart(contributions)
    
    is_high = prediction == 1
    color = "#DC2626" if is_high else "#059669"
    level = "HIGH RISK" if is_high else "LOW RISK"
    icon = "⚠" if is_high else "✓"
    
    html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; padding: 1.5rem;">
        <div style="font-size: 4rem; color: {color};">{icon}</div>
        <div style="font-size: 0.85rem; font-weight: 600; color: {color}; 
                    text-transform: uppercase; letter-spacing: 1px;">{level}</div>
        <div style="font-size: 2.5rem; font-weight: 700; color: {color};">{proba:.1%}</div>
        <div style="font-size: 0.8rem; color: #64748B;">Infection Probability</div>
    </div>
    """
    
    return html, fig

def load_random():
    """Load random case"""
    vals = get_random_sample()
    return vals[0], vals[1], vals[2], vals[3], vals[4], f"<b>Label:</b> {vals[5]}"

def reset():
    """Reset all inputs to defaults"""
    return 20.0, 50.0, 200.0, 5.0, 20.0, '<span class="label-tag">Load a random case</span>'

# CSS Styles
CSS = """
.gradio-container { background: #F8FAFC !important; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }

.header { background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%); padding: 2rem 1.5rem; text-align: center; border-radius: 16px; margin-bottom: 2rem; }
.header h1 { color: white; font-size: 1.75rem; font-weight: 600; margin: 0 0 0.5rem 0; }
.header p { color: rgba(255,255,255,0.75); font-size: 0.9rem; margin: 0; }

.flex-row, .flex-row > div, .flex-row > div > div { display: flex !important; align-items: stretch !important; }
.flex-row > div, .flex-row > div > div { flex-direction: row !important; }
.flex-row > div > div { flex-direction: column !important; }

.left-card, .right-card { background: white; border-radius: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #E2E8F0; padding: 1.5rem; height: 100% !important; min-height: 100% !important; }
.left-card > div, .right-card > div { height: 100% !important; }

.card-title { font-size: 0.85rem; font-weight: 600; color: #334155; margin-bottom: 1.25rem; text-transform: uppercase; letter-spacing: 0.5px; padding-bottom: 0.75rem; border-bottom: 1px solid #E2E8F0; }

input[type="number"] { border: 1px solid #CBD5E1 !important; border-radius: 10px !important; font-size: 1rem !important; padding: 0.65rem 0.85rem !important; background: #F8FAFC !important; transition: all 0.2s !important; color: #1E293B !important; }
input[type="number"]:focus { border-color: #3B82F6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important; background: white !important; }

.btn-analyze { background: #1E3A8A !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; padding: 0.75rem 1.5rem !important; width: 100%; transition: all 0.2s !important; }
.btn-analyze:hover { background: #1E40AF !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(30,58,138,0.25) !important; }

.btn-secondary { background: #F1F5F9 !important; color: #475569 !important; border: 1px solid #E2E8F0 !important; border-radius: 10px !important; font-weight: 500 !important; padding: 0.55rem 1rem !important; transition: all 0.2s !important; }
.btn-secondary:hover { background: #E2E8F0 !important; }

.result-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem 1.5rem; background: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 12px; color: #94A3B8; text-align: center; }
.label-tag { display: inline-block; padding: 0.35rem 0.9rem; background: #F1F5F9; color: #64748B; border-radius: 6px; font-size: 0.8rem; font-weight: 500; }

.legend { display: flex; justify-content: center; gap: 1.5rem; padding-top: 1rem; margin-top: 1rem; border-top: 1px solid #E2E8F0; }
.legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; color: #64748B; }
.legend-dot { width: 10px; height: 10px; border-radius: 3px; }

.footer { text-align: center; color: #94A3B8; font-size: 0.75rem; margin-top: 2rem; padding: 1rem 0; }
label { font-weight: 500 !important; color: #475569 !important; font-size: 0.85rem !important; }
.contain { padding: 0 !important; }
"""

# Build UI
with gr.Blocks(css=CSS, title="SBE Risk Assessment", theme=gr.themes.Default()) as demo:
    gr.HTML('<div class="header"><h1>SBE Risk Assessment System</h1><p>AI-Powered Infection Risk Prediction</p></div>')
    
    with gr.Row(elem_classes="flex-row"):
        with gr.Column(scale=4, elem_classes="left-card"):
            gr.HTML('<div class="card-title">Patient Parameters</div>')
            
            with gr.Row():
                random_btn = gr.Button("Load a Case", elem_classes="btn-secondary", scale=2)
                reset_btn = gr.Button("Reset", elem_classes="btn-secondary", scale=1)
                real_label = gr.HTML('<span class="label-tag">-</span>', scale=1)
            
            gr.HTML('<div style="height: 0.5rem;"></div>')
            
            inputs = [
                gr.Number(label="PMN(%)", value=20.0, precision=1),
                gr.Number(label="PMN", value=50.0, precision=1),
                gr.Number(label="Ascitic fluid WBC", value=200.0, precision=1),
                gr.Number(label="WBC", value=5.0, precision=1),
                gr.Number(label="CRP", value=20.0, precision=1)
            ]
            
            gr.HTML('<div style="height: 0.75rem;"></div>')
            predict_btn = gr.Button("Analyze", elem_classes="btn-analyze")
        
        with gr.Column(scale=7, elem_classes="right-card"):
            gr.HTML('<div class="card-title">Prediction Result</div>')
            result = gr.HTML('<div class="result-placeholder"><div style="font-size: 2.5rem; opacity: 0.5;">●</div><div style="font-size: 0.85rem;">Click "Analyze" to see prediction</div></div>')
            
            gr.HTML('<div style="height: 1.25rem;"></div>')
            gr.HTML('<div class="card-title" style="margin-top: 0.5rem;">Feature Contributions</div>')
            chart = gr.Plot(label="", show_label=False)
            
            gr.HTML('<div class="legend"><div class="legend-item"><div class="legend-dot" style="background: #3B82F6;"></div><span>Increases risk</span></div><div class="legend-item"><div class="legend-dot" style="background: #EF4444;"></div><span>Decreases risk</span></div></div>')
    
    gr.HTML('<div class="footer">For research purposes only</div>')
    
    random_btn.click(fn=load_random, outputs=[*inputs, real_label])
    reset_btn.click(fn=reset, outputs=[*inputs, real_label])
    predict_btn.click(fn=predict, inputs=inputs, outputs=[result, chart])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
