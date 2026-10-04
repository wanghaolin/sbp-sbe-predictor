#!/usr/bin/env python3
"""SBE Clinical Prediction System - Gradio Interface.

Serves the NAM-TabR predictor (src/models/nam_tabr.py) trained by
train_sbe_model.py as an interactive risk-assessment page with a
per-feature contribution chart.
"""

import os
import sys
import random
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# Path setup
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

from models.nam_tabr import NAMTabRClassifier

# Constants
FLUID = 'sbe'
PORT = 7861
FEATURE_NAMES = ['PMN%', 'PMN', 'Hydrothorax WBC', 'WBC', 'CRP']

# Illustrative example inputs for the "Load a Case" button (typical lab
# values, feature order = FEATURE_NAMES; replace with your own examples).
EXAMPLE_CASES = [
    [16.0, 53.9, 370.0, 4.1, 21.4],
    [3.0, 7.8, 260.0, 11.6, 21.4],
    [26.0, 74.4, 286.0, 26.5, 20.0],
]
DEFAULT_INPUTS = EXAMPLE_CASES[0]

# Load the trained predictor bundle produced by train_sbe_model.py.
# The bundle is self-contained: network weights, LR baseline, fitted scaler
# and the Youden-J decision threshold.
BUNDLE_PATH = os.path.join(project_dir, 'checkpoints', FLUID, 'model.pt')
if not os.path.exists(BUNDLE_PATH):
    raise SystemExit(
        f'Model bundle not found: {BUNDLE_PATH}\n'
        'Train it first with:  python train_sbe_model.py')
model = NAMTabRClassifier.load(BUNDLE_PATH)
scaler = model.scaler
THRESHOLD = float(model.best_threshold or 0.5)


# Clinical palette: red = pushes risk up, steel blue = pushes risk down
C_UP = '#DC2626'
C_DOWN = '#0284C7'
C_HIGH = '#DC2626'
C_LOW = '#059669'

def get_random_sample():
    return list(random.choice(EXAMPLE_CASES))

def create_chart(contributions):
    """Create feature contribution bar chart"""
    fig, ax = plt.subplots(figsize=(9.5, 4.1), dpi=110)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    colors = [C_UP if c >= 0 else C_DOWN for c in contributions]
    bars = ax.barh(FEATURE_NAMES, contributions, color=colors, height=0.58,
                   edgecolor='none', alpha=0.95)

    ax.axvline(0, color='#9AA1AB', linewidth=1.0)
    ax.set_xlabel('Contribution to log-odds', fontsize=15, fontweight='bold', color='#5B6470', labelpad=10)
    ax.tick_params(left=False, bottom=False, labelsize=17, colors='#111827')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold')

    lim = max(0.05, max(abs(c) for c in contributions) * 1.3)
    for bar, c in zip(bars, contributions):
        ax.text(bar.get_width() + (0.02 * lim if c >= 0 else -0.02 * lim),
                bar.get_y() + bar.get_height() / 2, f'{c:+.2f}',
                ha='left' if c >= 0 else 'right', va='center',
                fontsize=15, fontweight='bold', color='#404852', family='monospace')

    ax.set_xlim(-lim, lim)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#E2E6EA')
    ax.grid(axis='x', linestyle=':', linewidth=0.8, alpha=0.8, color='#E5E9EE')
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.6)
    return fig

def predict(pmn_ratio, pmn, wbc_fluid, wbc, crp):
    """Generate prediction and explanation"""
    X = np.array([[pmn_ratio, pmn, wbc_fluid, wbc, crp]], dtype=np.float32)
    X_scaled = scaler.transform(X)

    proba = float(model.predict_proba(X_scaled)[0, 1])
    prediction = int(proba >= THRESHOLD)

    # Per-feature gated NAM contributions to the log-odds
    contribs = model.get_feature_contributions(X_scaled)  # list per feature, each [n_samples]
    contributions = [float(c[0]) for c in contribs]
    fig = create_chart(contributions)

    is_high = prediction == 1
    color = C_HIGH if is_high else C_LOW
    level = "HIGH RISK" if is_high else "LOW RISK"

    html = f"""
    <div style="border-top: 3px solid {color}; padding: 1.5rem 1rem 1.3rem; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 0.6rem; margin-bottom: 0.6rem;">
            <span style="width: 11px; height: 11px; background: {color};"></span>
            <span style="font-size: 20px; font-weight: 700; color: {color}; letter-spacing: 4px;">{level}</span>
        </div>
        <div style="font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
                    font-size: 86px; font-weight: 700; color: #111827; line-height: 1.05;">{proba:.1%}</div>
    </div>
    """

    return html, fig

def load_random():
    """Load random case"""
    vals = get_random_sample()
    return (vals[0], vals[1], vals[2], vals[3], vals[4])

def reset():
    """Reset all inputs to defaults"""
    return (16.0, 53.9, 370.0, 4.1, 21.4)

# CSS Styles
CSS = """
:root {
    --body-background-fill: #FAFBFC;
    --block-background-fill: #FFFFFF;
    --block-border-color: #E2E6EA;
    --input-background-fill: #FFFFFF;
    --input-border-color: #D3D9DF;
    --body-text-color: #111827;
    --block-label-text-color: #404852;
}

.gradio-container {
    background: #FAFBFC !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    max-width: 1180px !important;
    color: #111827;
}

/* ---------- masthead ---------- */
.masthead {
    display: flex; align-items: baseline; justify-content: space-between;
    padding: 2.1rem 0.4rem 1.5rem; border-bottom: 1px solid #E2E6EA; margin-bottom: 2rem;
}
.masthead h1 { font-size: 50px; font-weight: 700; letter-spacing: -0.8px; color: #111827; margin: 0; line-height: 1.15; }
.masthead-chip {
    font-family: Consolas, 'Courier New', monospace; font-size: 14px; font-weight: 700;
    letter-spacing: 2.5px; color: #0891B2; text-align: right; line-height: 1.6; font-size: 15px;
    white-space: nowrap;
}
.masthead-chip span { font-size: 10.5px; font-weight: 600; color: #9AA1AB; letter-spacing: 2px; }

/* ---------- layout / cards ---------- */
.flex-row, .flex-row > div, .flex-row > div > div { display: flex !important; align-items: stretch !important; }
.flex-row > div, .flex-row > div > div { flex-direction: row !important; }
.flex-row > div > div { flex-direction: column !important; }

.left-card, .right-card {
    background: #FFFFFF; border: 1px solid #E2E6EA; border-radius: 8px;
    box-shadow: none; padding: 1.6rem 1.7rem; height: 100% !important; min-height: 100% !important;
}
.left-card > div, .right-card > div { height: 100% !important; }

.card-title {
    font-family: Consolas, 'Courier New', monospace; font-size: 13px; font-weight: 700;
    letter-spacing: 2px; color: #5B6470; text-transform: uppercase; font-size: 17px;
    margin-bottom: 1.3rem; padding-bottom: 0.7rem; border-bottom: 1px solid #E2E6EA;
}

/* ---------- form ---------- */
.gradio-container label, .gradio-container label span {
    font-weight: 700 !important; color: #404852 !important; font-size: 19px !important;
}
input[type="number"] {
    border: 1px solid #D3D9DF !important; border-radius: 6px !important;
    font-family: Consolas, 'Courier New', monospace !important;
    font-size: 21px !important; font-weight: 700 !important; padding: 0.75rem 0.95rem !important;
    background: #FFFFFF !important; color: #111827 !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
input[type="number"]:focus {
    border-color: #0891B2 !important;
    box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.14) !important;
}

/* ---------- buttons ---------- */
.btn-analyze {
    background: #111827 !important; color: #FFFFFF !important; border: none !important;
    border-radius: 6px !important; font-size: 19px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    padding: 0.85rem 1.5rem !important; width: 100%; cursor: pointer !important;
}
.btn-analyze:hover { background: #0891B2 !important; }

.btn-secondary {
    background: #FFFFFF !important; color: #404852 !important;
    border: 1px solid #D3D9DF !important; border-radius: 6px !important;
    font-size: 17.5px !important; font-weight: 700 !important; padding: 0.65rem 1.05rem !important;
}
.btn-secondary:hover { border-color: #0891B2 !important; color: #0891B2 !important; }

/* ---------- result ---------- */
.result-placeholder {
    display: flex; align-items: center; justify-content: center;
    padding: 2.8rem 1.5rem; border: 1px dashed #D3D9DF; border-radius: 6px;
    color: #9AA1AB; font-size: 18px; font-weight: 600; text-align: center;
}
.label-tag {
    display: inline-block; padding: 0.4rem 0.9rem; border: 1px solid #D3D9DF;
    border-radius: 6px; font-family: Consolas, 'Courier New', monospace;
    font-size: 16px; font-weight: 700; color: #404852; background: #FFFFFF;
}
.label-tag-muted { color: #9AA1AB; }

/* ---------- legend / footer ---------- */
.legend {
    display: flex; justify-content: center; gap: 2.2rem;
    padding-top: 1.1rem; margin-top: 1.2rem; border-top: 1px solid #E2E6EA;
}
.legend-item { display: flex; align-items: center; gap: 0.55rem; font-size: 17px; font-weight: 600; color: #5B6470; }
.legend-dot { width: 13px; height: 13px; border-radius: 2px; }

.footer {
    text-align: center; font-family: Consolas, 'Courier New', monospace;
    color: #9AA1AB; font-size: 14px; letter-spacing: 1px;
    margin-top: 2.2rem; padding: 1.1rem 0 0.6rem; border-top: 1px solid #E2E6EA;
}
.contain { padding: 0 !important; }
footer { display: none !important; }
"""

# Build UI
with gr.Blocks(title="SBE Risk Assessment") as demo:
    gr.HTML(
        '<div class="masthead">'
        '<h1>SBE Risk Assessment</h1>'
        '<div class="masthead-chip">NAM-TABR<br><span>HYBRID MODEL</span></div>'
        '</div>'
    )

    with gr.Row(elem_classes="flex-row"):
        with gr.Column(scale=4, elem_classes="left-card"):
            gr.HTML('<div class="card-title">Parameters</div>')

            with gr.Row():
                random_btn = gr.Button("Load a Case", elem_classes="btn-secondary", scale=2)
                reset_btn = gr.Button("Reset", elem_classes="btn-secondary", scale=1)

            gr.HTML('<div style="height: 0.6rem;"></div>')

            inputs = [
                gr.Number(label="PMN(%)", value=16.0),
                gr.Number(label="PMN", value=53.9),
                gr.Number(label="Hydrothorax WBC", value=370.0),
                gr.Number(label="WBC", value=4.1),
                gr.Number(label="CRP", value=21.4)
            ]

            gr.HTML('<div style="height: 0.9rem;"></div>')
            predict_btn = gr.Button("Analyze", elem_classes="btn-analyze")

        with gr.Column(scale=7, elem_classes="right-card"):
            gr.HTML('<div class="card-title">Prediction</div>')
            result = gr.HTML('<div class="result-placeholder">Enter parameters, then Analyze</div>')

            gr.HTML('<div style="height: 1.4rem;"></div>')
            gr.HTML('<div class="card-title">Feature Contributions</div>')
            chart = gr.Plot(label="", show_label=False)

            gr.HTML(
                '<div class="legend">'
                f'<div class="legend-item"><div class="legend-dot" style="background: {C_UP};"></div><span>Raises risk</span></div>'
                f'<div class="legend-item"><div class="legend-dot" style="background: {C_DOWN};"></div><span>Lowers risk</span></div>'
                '</div>'
            )

    gr.HTML('<div class="footer">For research purposes only</div>')

    random_btn.click(fn=load_random, outputs=inputs)
    reset_btn.click(fn=reset, outputs=inputs)
    predict_btn.click(fn=predict, inputs=inputs, outputs=[result, chart])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, css=CSS, theme=gr.themes.Default())
