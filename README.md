# Clinial Risk Assessment System

A deep learning-based infection risk prediction system, including two prediction models for SBE and SBP.

## Project Structure

```
├── app_gradio_sbe.py      # SBE Risk Assessment Web Interface
├── app_gradio_sbp.py      # SBP Risk Assessment Web Interface
├── train_sbe_model.py     # SBE Model Training Script
├── train_sbp_model.py     # SBP Model Training Script
├── src/                   # Source Code Directory
│   ├── data/              # Data Loading Module
│   └── models/            # Model Definition Module
├── data/                  # Training Data Directory
└── temp/                  # Models and Intermediate Results Directory
```

## Requirements

- Python 3.8+
- PyTorch
- Gradio
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

Install dependencies:
```bash
pip install torch gradio scikit-learn pandas numpy matplotlib joblib
```

## Usage

### 1. Model Training

Train SBE model:
```bash
python train_sbe_model.py
```

Train SBP model:
```bash
python train_sbp_model.py
```

After training, model files are saved in the `temp/models/` directory.

### 2. Launch Prediction Interface

Launch SBE risk assessment interface (port 7860):
```bash
python app_gradio_sbe.py
```

Launch SBP risk assessment interface (port 7860):
```bash
python app_gradio_sbp.py
```

After launching, visit `http://localhost:7860` to access the web interface.

## Model Description

### SBE Model
- Input Features: PMN%, PMN, Ascitic fluid WBC, WBC, CRP
- Core Features: PMN%, PMN

### SBP Model
- Input Features: PMN, Ascitic fluid WBC, Total cell count, Lymphocyte percentage
- Core Features: PMN

## Output Description

- Risk Level: HIGH RISK / LOW RISK
- Feature Contribution Chart: Shows the contribution of each feature to the prediction result

## Notes

- Training data needs to be prepared in advance and placed in the `data/` directory
- First-time use requires running the training scripts to generate model files
