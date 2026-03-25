# BRA Logger — AI-Powered Breast Anomaly Risk Scoring System

An IoT wearable system that uses embedded temperature, pressure, and bioimpedance sensors to capture breast tissue data, process it through an AI/ML pipeline, and generate anomaly risk scores — assisting in the early identification of abnormal breast tissue patterns.

> **Current status:** Phase 1 (data simulation) and Phase 2 (ML pipeline) are complete.  
> Phases 3–6 (backend API, frontend dashboard, firmware, deployment) are pending.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Phase 1 — Generate Simulated Data](#phase-1--generate-simulated-data)
4. [Phase 2 — ML Pipeline](#phase-2--ml-pipeline)
5. [What Gets Created](#what-gets-created)
6. [Switching Algorithms](#switching-algorithms)
7. [Project Structure](#project-structure)

---

## Prerequisites

| Tool | Minimum Version | Notes |
|---|---|---|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Git | Any | [git-scm.com](https://git-scm.com) |
| pip | Bundled with Python | — |

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jonathanmuk/Bra-Logger.git
cd bra_logger
```

---

### 2. Create a Virtual Environment

**Windows (PowerShell)**
```powershell
python -m venv venv
```

**macOS / Linux**
```bash
python3 -m venv venv
```

---

### 3. Activate the Virtual Environment

**Windows (PowerShell)**
```powershell
.\venv\Scripts\Activate.ps1
```

> If you get an execution policy error on Windows, run this first (once):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux**
```bash
source venv/bin/activate
```

Your terminal prompt will show `(venv)` when the environment is active.

---

### 4. Install Dependencies

**Windows & macOS / Linux**
```bash
pip install -r requirements.txt
```

This installs all required packages: numpy, pandas, scikit-learn, xgboost, shap, optuna, mlflow, and more.

> **Optional — PyTorch for deep learning models (Autoencoder, MLP):**
>
> **Windows / Linux (CPU only)**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
>
> **macOS**
> ```bash
> pip install torch
> ```

---

## Phase 1 — Generate Simulated Data

The repository does **not** include data files — they are generated locally. This keeps the repo lightweight and lets each contributor generate their own training data.

**Windows & macOS / Linux**
```bash
python -m data_collection.simulate_data
```

**What happens:**
- Generates **500 healthy sessions** → `ml/data/raw/simulated/healthy/`
- Generates **150 anomaly sessions** → `ml/data/raw/simulated/anomaly/`
- Each session is a CSV file with ~600 rows of sensor readings (5 minutes at 2 Hz)
- Anomaly sessions include injected patterns: temperature hotspots, pressure asymmetry, impedance shifts

**Expected output:**
```
Generating healthy sessions: 100%|████████████| 500/500
Generating anomaly sessions: 100%|████████████| 150/150
Generated 500 healthy + 150 anomaly sessions
```

**Total data:** 650 CSV files, ~35,750 windows of 30-second segments.

---

## Phase 2 — ML Pipeline

> **Requires:** Phase 1 data to exist. Run `simulate_data` first.

All commands below must be run from the project root with the virtual environment active.

---

### Default Run (recommended first run)

**Windows & macOS / Linux**
```bash
python -m ml.run_pipeline
```

Trains both models and generates evaluation artifacts. Takes ~15–20 minutes depending on hardware.

**Steps performed:**
1. Loads 650 sessions from `ml/data/raw/simulated/`
2. Creates 35,750 sliding windows (30s window, 5s step, 2 Hz)
3. Extracts 389 features per window (time-domain, spectral, spatial, cross-modal)
4. Splits 80% train / 20% test (stratified)
5. Trains **IsolationForest** (anomaly detector — unsupervised, healthy data only)
6. Trains **XGBClassifier** (supervised classifier — uses both classes)
7. Evaluates both on the test set and generates plots
8. Computes SHAP feature importance explanations

---

### Anomaly Detector Only

```bash
python -m ml.run_pipeline --mode anomaly
```

---

### Classifier Only

```bash
python -m ml.run_pipeline --mode classifier
```

---

### Skip SHAP Explanations (faster)

```bash
python -m ml.run_pipeline --no-explain
```

---

### With Optuna Hyperparameter Tuning

```bash
python -m ml.run_pipeline --tune
```

Runs Bayesian optimization (50 trials, 5-minute timeout) before training each model to find the best hyperparameters. Adds ~5–15 minutes.

---

### With MLflow Experiment Tracking

```bash
python -m ml.run_pipeline --track
```

Logs all parameters, metrics, and artifact paths to MLflow. Creates a local `mlruns/` directory.

After running, launch the MLflow UI to browse results:

**Windows & macOS / Linux**
```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

### All Features Combined

```bash
python -m ml.run_pipeline --tune --track
```

---

## What Gets Created

After running the pipeline, the following files are generated locally (not tracked in git — regenerated each run):

```
ml/models/simulation/
├── anomaly_model.joblib          ← trained IsolationForest (or other anomaly algorithm)
├── scaler.joblib                 ← StandardScaler fitted to healthy training windows
├── feature_names.json            ← ordered list of 389 feature names
├── classifier_model.joblib       ← trained XGBClassifier (or other classifier)
├── classifier_scaler.joblib      ← StandardScaler fitted to all training windows
├── classifier_feature_names.json ← feature names for classifier
└── evaluation/
    ├── anomaly_metrics.json          ← AUC-ROC, F1, precision, recall
    ├── classifier_metrics.json
    ├── anomaly_roc_curve.png         ← ROC curve
    ├── anomaly_pr_curve.png          ← Precision-Recall curve
    ├── anomaly_confusion_matrix.png  ← Confusion matrix heatmap
    ├── anomaly_score_distribution.png← Score histogram (healthy vs. anomaly)
    ├── anomaly_shap_summary.png      ← SHAP beeswarm plot
    ├── anomaly_shap_bar.png          ← Top-20 features by importance
    ├── anomaly_feature_importance.json
    ├── classifier_roc_curve.png
    ├── classifier_pr_curve.png
    ├── classifier_confusion_matrix.png
    ├── classifier_score_distribution.png
    ├── classifier_shap_summary.png
    ├── classifier_shap_bar.png
    └── classifier_feature_importance.json
```

If `--track` was used, MLflow also creates:
```
mlruns/                           ← MLflow experiment tracking (not tracked in git)
```

---

## Switching Algorithms

The pipeline supports pluggable algorithms — no code changes needed. Edit `ml/config/ml_config.yaml`:

```yaml
# Anomaly detector options:
# IsolationForest | OneClassSVM | LocalOutlierFactor | Autoencoder (requires torch)
anomaly_model:
  algorithm: "IsolationForest"

# Classifier options:
# XGBClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier (requires torch)
classifier:
  algorithm: "XGBClassifier"
```

Then re-run any pipeline command — the new algorithm is automatically picked up.

---

## Project Structure

```
bra_logger/
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
├── platformio.ini                   ← Firmware build config
├── implementation.md                ← Full system implementation plan
├── phase2_implementation.md         ← Detailed ML pipeline documentation
│
├── data_collection/                 ← Phase 1: data collection & simulation
│   ├── config.yaml                  ← Sensor channel definitions & sim parameters
│   ├── simulate_data.py             ← Generates simulated healthy + anomaly sessions
│   ├── receiver_logger.py           ← Records live data from hardware over BLE/serial
│   ├── session_manager.py           ← Session directory management
│   └── schema.py                    ← CSV schema validation
│
├── ml/                              ← Phase 2: ML pipeline
│   ├── run_pipeline.py              ← CLI entry point (--mode, --tune, --track, --no-explain)
│   ├── config/
│   │   └── ml_config.yaml           ← All ML hyperparameters and settings
│   ├── data/
│   │   └── raw/
│   │       ├── simulated/
│   │       │   ├── healthy/         ← 500 healthy CSVs (gitignored, regenerate locally)
│   │       │   └── anomaly/         ← 150 anomaly CSVs (gitignored, regenerate locally)
│   │       └── hardware/
│   │           ├── healthy/         ← future real hardware data
│   │           └── clinical/        ← future clinical validation data
│   ├── models/                      ← Trained models & evaluation (gitignored)
│   ├── pipelines/
│   │   ├── preprocess.py            ← Load & validate raw CSVs
│   │   ├── feature_engineering.py   ← Extract 389 features per window
│   │   ├── model_registry.py        ← Pluggable algorithm factory
│   │   ├── dl_models.py             ← PyTorch DL model wrappers (optional)
│   │   ├── train_anomaly.py         ← Anomaly detector training
│   │   ├── train_classifier.py      ← Classifier training
│   │   ├── tuning.py                ← Optuna hyperparameter tuning
│   │   ├── tracking.py              ← MLflow experiment tracking
│   │   ├── evaluate.py              ← Metrics + diagnostic plots
│   │   └── explain.py               ← SHAP explainability
│   └── utils/
│       ├── features.py              ← Low-level feature functions
│       ├── visualization.py         ← Reusable plot helpers
│       └── windowing.py             ← Sliding window utilities
│
├── firmware/                        ← Phase 1: ESP32 firmware (PlatformIO)
│   └── src/
│       ├── config.h
│       ├── communication/
│       ├── sensors/
│       └── utils/
│
└── src/                             ← Legacy firmware entry point
    └── main.cpp
```

---

## Notes

- **Simulated data is not committed to git.** Each contributor runs `python -m data_collection.simulate_data` to generate their own copy. This keeps the repo small and the pipeline auditable.
- **Trained models are not committed to git.** Run `python -m ml.run_pipeline` to regenerate them. Model artifacts are deterministic given the same data and random seeds.
- **MLflow tracking runs are not committed to git.** Use `mlflow ui` to browse them locally.
- **PyTorch is optional.** All ML algorithms work without it. PyTorch is only needed if you set the algorithm to `Autoencoder` or `MLPClassifier` in `ml_config.yaml`.
