# Phase 2 Implementation — ML Pipeline

> **Status:** Code complete (enhanced with pluggable model registry, Optuna tuning, and MLflow tracking).  
> **Depends on:** Phase 1 (simulated data generated — 500 healthy + 150 anomaly sessions).  
> **Hardware required:** None — runs entirely on the CSV data from Phase 1.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [What Was Built and Why](#what-was-built-and-why)
3. [File-by-File Walkthrough](#file-by-file-walkthrough)
4. [Data Flow — End to End](#data-flow--end-to-end)
5. [Feature Engineering Details](#feature-engineering-details)
6. [Models](#models)
7. [Pluggable Model Registry](#pluggable-model-registry)
8. [Hyperparameter Tuning (Optuna)](#hyperparameter-tuning-optuna)
9. [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
10. [Evaluation & Metrics](#evaluation--metrics)
11. [Explainability (SHAP)](#explainability-shap)
12. [Configuration Reference](#configuration-reference)
13. [How to Run](#how-to-run)
14. [Folder Structure](#folder-structure)

---

## High-Level Overview

**Analogy — The Quality Inspector:**

Imagine you run a factory that produces ceramic tiles. Most tiles come out fine, but occasionally one has a hidden crack. You hire a quality inspector.

- **Phase 1** was about building the factory floor — the machines that stamp out tiles (the data simulator) and the conveyor belt that moves them into storage (the CSV files).
- **Phase 2** is about hiring and training the inspector.

The inspector works in three steps:

1. **Look at small sections** (windowing) — instead of staring at an entire 5-minute recording at once, she examines 30-second slices, checking each one independently.
2. **Measure key attributes** (feature engineering) — she doesn't memorize every raw number. Instead, she measures summary statistics: "Is this section hotter than average? Is the left side different from the right? Is there a sudden spike?"
3. **Learn what normal looks like** (training) — by studying hundreds of healthy tiles, she builds an intuition for "normal." Anything that falls far outside that pattern gets flagged.

Phase 2 builds exactly this pipeline in code: windows → features → models → evaluation → explanations.

---

## What Was Built and Why

| Component | File | Purpose |
|---|---|---|
| ML Configuration | `ml/config/ml_config.yaml` | Central config for windowing, features, model hyperparameters, tuning, tracking, output paths |
| Feature Engineering | `ml/pipelines/feature_engineering.py` | Converts raw sensor windows into ~389 numeric features per window |
| **Model Registry** | `ml/pipelines/model_registry.py` | **Pluggable algorithm factory — swap any ML/DL model via config** |
| **Deep Learning Models** | `ml/pipelines/dl_models.py` | **PyTorch wrappers (Autoencoder, MLP) with sklearn-compatible interface** |
| Anomaly Detection | `ml/pipelines/train_anomaly.py` | Trains anomaly detector (any registered algorithm) on healthy data |
| Supervised Classifier | `ml/pipelines/train_classifier.py` | Trains classifier (any registered algorithm) on labelled data |
| **Hyperparameter Tuning** | `ml/pipelines/tuning.py` | **Optuna-based hyperparameter optimization for any registered model** |
| **Experiment Tracking** | `ml/pipelines/tracking.py` | **MLflow wrapper — logs params, metrics, artifacts, models** |
| Evaluation | `ml/pipelines/evaluate.py` | Computes metrics (AUC-ROC, F1, precision, recall) + generates plots |
| Explainability | `ml/pipelines/explain.py` | SHAP values — shows which features drive each prediction |
| Pipeline Runner | `ml/run_pipeline.py` | CLI entry point — orchestrates the full pipeline with --tune and --track flags |

### Why two model categories?

The pipeline supports two complementary model categories, each of which can use **any registered algorithm** from the model registry:

1. **Anomaly Detector (unsupervised/semi-supervised):** Trained on healthy data only. Learns "what normal looks like" and flags anything sufficiently different. Default: Isolation Forest. Alternatives: OneClassSVM, LocalOutlierFactor, Autoencoder (PyTorch).

2. **Classifier (supervised):** Trained on both healthy and anomaly data with labels. More accurate when labels are available. Default: XGBClassifier. Alternatives: RandomForestClassifier, GradientBoostingClassifier, SVC, MLPClassifier (PyTorch).

**Switching algorithms is a one-line config change:**
```yaml
anomaly_model:
  algorithm: "OneClassSVM"    # was "IsolationForest"
```

**Analogy:** The model registry is like a universal remote — instead of buying a separate remote for each TV, you have one remote that works with any brand. The pipeline is the remote; algorithms are TVs you can swap freely.

---

### Modular Architecture

The enhanced pipeline follows a **plug-and-play** architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                     ml_config.yaml                                │
│   algorithm: "XGBClassifier"  ←  change this to swap models      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Model Registry (model_registry.py)                   │
│   IsolationForest | OneClassSVM | LOF | XGBoost | RF | SVC       │
│   Autoencoder (DL) | MLPClassifier (DL)                          │
│   Each entry: builder + default_params + optuna_search_space      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         train_*.py  tuning.py  tracking.py
         (uses registry) (Optuna)  (MLflow)
```

---

## File-by-File Walkthrough

### `ml/config/ml_config.yaml`

**What it does:** Centralizes all tunable parameters so you never have to dig through Python code to change a setting.

**Key sections:**
- `windowing` — Window size (30s), step (5s), sampling rate (2 Hz)
- `features` — Toggle feature families on/off
- `anomaly_model` — Isolation Forest hyperparameters (200 trees, 5% contamination)
- `classifier` — XGBoost hyperparameters (200 trees, depth 6, learning rate 0.1)
- `split` — 80/20 train/test split with stratification
- `evaluation` — Score threshold (0.5), which plots to generate
- `explain` — SHAP subsample size (200), top-N features to show (20)
- `output` — Where to save models and evaluation artifacts

**Analogy:** This is the control panel for the entire factory — you adjust knobs here instead of rewiring machines.

---

### `ml/pipelines/feature_engineering.py`

**What it does:** Takes a list of 30-second window DataFrames and distills each one into ~150 numbers that describe what happened in that window.

**How it works:**

For each window, it computes four families of features across all 21 sensor channels:

1. **Time-domain features** (13 per channel) — Statistical summaries: mean, std, min, max, range, median, IQR, skewness, kurtosis, linear slope, signal energy, first-derivative mean & std.

2. **Spectral features** (4 per channel) — Frequency-domain analysis via FFT: dominant frequency, spectral entropy, spectral energy, spectral centroid.

3. **Spatial features** — Cross-channel comparisons:
   - Left vs. right breast differences (6 features × 3 modalities = 18 features)
   - Hotspot detection (4 features × 3 modalities = 12 features)
   - Cross-modal correlations (temperature vs. pressure, temperature vs. impedance = 2 features)

4. **Label extraction** — Binary label (0 or 1) from the window's `label` column.

**Why these features?**

Each feature family captures a different type of anomaly:
- **Time-domain** catches sudden spikes, unusual variance, or trending values
- **Spectral** catches periodic patterns (e.g. rhythmic muscle contractions vs steady tissue)
- **Left-right differences** catch asymmetry — the hallmark of unilateral pathology
- **Hotspot** catches localised extremes — a single sensor reading much hotter than the rest

**Analogy:** Imagine describing a song to someone who can't hear it. You wouldn't give them every sound wave sample — you'd say "it's loud, fast, in a minor key, with a guitar solo at the 2-minute mark." Features are that summary.

**Key function:** `extract_features(windows) → (feature_df, labels)`

---

### `ml/pipelines/train_anomaly.py`

**What it does:** Trains an anomaly detector using any algorithm from the model registry, and provides scoring functions.

**How it works:**

The training function reads the algorithm name from config (e.g. `"IsolationForest"`, `"OneClassSVM"`, `"Autoencoder"`), looks it up in the model registry, and instantiates it with the configured or tuned parameters.

**Supported anomaly algorithms:**

| Algorithm | Type | When to use |
|---|---|---|
| IsolationForest | ML (sklearn) | Default — fast, robust, good baseline |
| OneClassSVM | ML (sklearn) | When data has clear boundaries |
| LocalOutlierFactor | ML (sklearn) | When density matters more than isolation |
| Autoencoder | DL (PyTorch) | When feature interactions are complex/non-linear |

**Training approach (semi-supervised):**
- We train on **healthy data only** from the training set
- At inference, the model scores windows from the test set
- Scores are normalised to [0, 1] where 1 = most anomalous

**Key functions:**
- `train_anomaly_detector(X_train)` → model, scaler, feature_names
- `score_anomalies(model, scaler, X)` → scores in [0, 1]
- `save_artifacts()` / `load_artifacts()` — persist/restore model to disk

**Outputs:**
- `anomaly_model.joblib` — the trained model
- `scaler.joblib` — the StandardScaler (features must be scaled the same way at inference)
- `feature_names.json` — ordered column names (so the backend knows what features to pass)

**Analogy:** The Isolation Forest is like a bouncer at an exclusive club. They've memorized what regular patrons look like. If someone shows up who doesn't fit the pattern, they get flagged — not because the bouncer knows what a "bad" person looks like, but because they know what a "normal" person looks like.

---

### `ml/pipelines/train_classifier.py`

**What it does:** Trains a supervised classifier using any algorithm from the model registry.

**Supported classifier algorithms:**

| Algorithm | Type | When to use |
|---|---|---|
| XGBClassifier | ML (xgboost) | Default — top accuracy, handles imbalanced data |
| RandomForestClassifier | ML (sklearn) | Robust, interpretable, no tuning needed |
| GradientBoostingClassifier | ML (sklearn) | Similar to XGBoost, pure sklearn |
| SVC | ML (sklearn) | Good for small datasets with clear margins |
| MLPClassifier | DL (PyTorch) | When deep non-linear boundaries needed |

**How it differs from the anomaly detector:**

| Aspect | Isolation Forest | XGBoost Classifier |
|---|---|---|
| Training data | Healthy only | Healthy + Anomaly (labelled) |
| Learning style | Unsupervised (learns "normal") | Supervised (learns boundary) |
| Outputs | Anomaly score [0,1] | Class probability [0,1] |
| When to use | No labels available | Labels available |
| Accuracy | Good | Usually better (with good labels) |

**Key features:**
- Algorithm is selected from config and resolved via model registry
- Supports `params_override` from Optuna tuning
- Automatic integration with MLflow tracking when enabled

**Analogy:** The old code was like a restaurant with a fixed menu — you could only order burger or pizza. The new code is like a restaurant with an open kitchen — name any dish (algorithm), and the chef (registry) knows how to make it.

---

### `ml/pipelines/evaluate.py`

**What it does:** Measures how well the models perform and generates diagnostic plots.

**Metrics computed:**
- **Accuracy** — Overall fraction of correct predictions
- **Precision** — Of windows flagged as anomaly, how many actually are?
- **Recall** — Of actual anomaly windows, how many did we catch?
- **F1 Score** — Harmonic mean of precision and recall (balanced single metric)
- **AUC-ROC** — Area Under the ROC Curve (how well scores separate classes)
- **AUC-PR** — Area Under the Precision-Recall Curve (better for imbalanced data)

**Plots generated:**
1. **ROC Curve** — True positive rate vs. false positive rate at all thresholds
2. **Precision-Recall Curve** — Precision vs. recall at all thresholds
3. **Confusion Matrix** — 2×2 grid of TP/FP/TN/FN counts
4. **Score Distribution** — Histogram of scores for healthy vs. anomaly windows

**Why these metrics?**

For a medical screening tool, **recall is critical** — we'd rather flag a healthy session for review (false positive) than miss a real anomaly (false negative). AUC-PR is particularly important because our data is imbalanced (~77% healthy, ~23% anomaly).

**Analogy:** Imagine testing a fire alarm. Accuracy tells you "it's usually right." Recall tells you "it catches 95% of actual fires." Precision tells you "only 10% of its alarms are false." For fire alarms (and medical screening), recall matters most.

**Key functions:**
- `evaluate_anomaly(scores, y_true)` → metrics dict
- `evaluate_classifier(y_prob, y_true)` → metrics dict

---

### `ml/pipelines/explain.py`

**What it does:** Uses SHAP (SHapley Additive exPlanations) to explain why the model made each prediction.

**Why explainability matters:**

In a medical context, a model that says "anomaly detected" is not enough. Clinicians need to know:
- *Which sensors* are driving the prediction?
- *Is it temperature asymmetry or pressure irregularity?*
- *Is it a single sensor spike or a systemic pattern?*

SHAP answers these questions by computing the marginal contribution of each feature to each prediction, based on game theory (Shapley values).

**Outputs:**
- **SHAP Summary Plot** — Beeswarm plot showing how each feature pushes predictions toward/away from anomaly across all test samples
- **SHAP Bar Plot** — Top-20 features ranked by mean absolute SHAP value
- **Feature Importance JSON** — Machine-readable ranked importance for downstream use

**Analogy:** When a doctor says "this X-ray looks abnormal," you want to know why. SHAP is like the doctor pointing at the X-ray and saying "see this shadow here, and this asymmetry there — those are the reasons."

---

### `ml/run_pipeline.py`

**What it does:** Single command to run the entire pipeline end-to-end, with optional tuning and tracking.

**Steps executed:**
1. Load healthy + anomaly CSV sessions from `ml/data/raw/simulated/`
2. Window them (30s windows, 5s step → ~55 windows per session)
3. Extract ~389 features per window
4. Stratified 80/20 train/test split
5. *(Optional)* Optuna hyperparameter tuning for each model
6. Train anomaly detector (from registry) on healthy-only training data
7. Train classifier (from registry) on all labelled training data
8. Evaluate both models on the held-out test set
9. Generate SHAP explanations for both models
10. *(Optional)* Log everything to MLflow
11. Save all models, metrics, and plots

**CLI options:**
- `--mode anomaly` — only train/evaluate the anomaly detector
- `--mode classifier` — only train/evaluate the classifier
- `--mode all` — both (default)
- `--no-explain` — skip SHAP explanations for faster runs
- `--tune` — enable Optuna hyperparameter tuning
- `--track` — enable MLflow experiment tracking

---

## Data Flow — End to End

```
650 CSV sessions (500 healthy + 150 anomaly)
        │
        ▼
┌──────────────────────┐
│   load_sessions()    │    Load & clean each CSV
│   (preprocess.py)    │    Validate columns, clip bounds, impute missing
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ windows_from_sessions│    30s windows @ 2 Hz = 60 rows each
│   (windowing.py)     │    5s step → ~55 windows/session
└──────────┬───────────┘    ~35,750 total windows
           │
           ▼
┌──────────────────────┐
│  extract_features()  │    ~389 features per window
│(feature_engineering) │    Time-domain + spectral + spatial + cross-modal
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   train_test_split   │    80% train / 20% test (stratified)
└───────┬──────┬───────┘
        │      │
   train set  test set
        │      │
        ▼      │
┌──────────────┐  │
│ Optuna Tuning│  │   ← optional (--tune flag)
│ (tuning.py)  │  │   Find optimal hyperparameters
└──────┬───────┘  │
       │          │
       ▼          │
┌──────────────┐  │
│ Train Models │  │   ← uses model registry (any algorithm)
│ - Anomaly    │  │
│ - Classifier │  │
└──────┬───────┘  │
       │          │
       ▼          ▼
┌──────────────────────┐
│     Evaluate on      │    AUC-ROC, F1, Precision, Recall
│     test set         │    ROC curves, PR curves, confusion matrices
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   SHAP Explanations  │    Feature importance rankings
│                      │    Summary plots, bar charts
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   MLflow Tracking    │    ← optional (--track flag)
│   Log all results    │    Params, metrics, artifacts, models
└──────────┬───────────┘
           │
           ▼
    ml/models/simulation/
    ├── anomaly_model.joblib
    ├── scaler.joblib
    ├── classifier_model.joblib
    ├── classifier_scaler.joblib
    ├── feature_names.json
    └── evaluation/
        ├── anomaly_metrics.json
        ├── classifier_metrics.json
        ├── *_roc_curve.png
        ├── *_pr_curve.png
        ├── *_confusion_matrix.png
        ├── *_score_distribution.png
        ├── *_shap_summary.png
        ├── *_shap_bar.png
        └── *_feature_importance.json
```

---

## Feature Engineering Details

### Feature Count Breakdown

| Category | Features per channel | Channels | Total |
|---|---|---|---|
| Time-domain | 13 | 21 (all sensors) | 273 |
| Spectral | 4 | 21 (all sensors) | 84 |
| L-R difference | 6 | 3 (temp, press, imp) | 18 |
| Hotspot | 4 | 3 (temp, press, imp) | 12 |
| Cross-modal correlation | 1 | 2 (temp⟷press, temp⟷imp) | 2 |
| **Total** | | | **~389** |

> **Note:** Actual count may be fewer if some channels are missing from a CSV. The feature matrix is filled with 0.0 for any missing features, ensuring consistent dimensionality.

### Time-Domain Features (per channel)

| Feature | Formula/Description | What it catches |
|---|---|---|
| `mean` | Mean of values | Baseline level |
| `std` | Standard deviation | Signal variability |
| `min` / `max` | Extremes | Spikes or drops |
| `range` | max − min | Total spread |
| `median` | 50th percentile | Robust central tendency |
| `iqr` | Q3 − Q1 | Spread without outlier influence |
| `skew` | Skewness | Asymmetry of distribution |
| `kurtosis` | Kurtosis | Tail heaviness (spikiness) |
| `slope` | Linear regression slope | Trend over window |
| `energy` | Mean of x² | Signal power |
| `d1_mean` | Mean of first derivative | Average rate of change |
| `d1_std` | Std of first derivative | Variability of rate of change |

### Spectral Features (per channel)

| Feature | Description | What it catches |
|---|---|---|
| `dominant_freq` | Frequency with highest FFT magnitude | Periodic patterns |
| `spectral_entropy` | Entropy of power spectrum | Randomness vs. periodicity |
| `spectral_energy` | Total power | Signal strength in frequency domain |
| `spectral_centroid` | Weighted average frequency | "Center of mass" of spectrum |

### Spatial Features

| Feature | Description |
|---|---|
| `*_lr_diff_mean` | Mean left-right difference (asymmetry indicator) |
| `*_lr_diff_std` | Variability of asymmetry over the window |
| `*_lr_diff_max` | Maximum instantaneous asymmetry |
| `*_lr_diff_min` | Minimum asymmetry (useful for sign) |
| `*_lr_asym_frac_05` | Fraction of window where \|L−R\| > 0.5 |
| `*_lr_asym_frac_1` | Fraction of window where \|L−R\| > 1.0 |
| `*_hotspot_max_deviation` | Max channel − mean of all channels |
| `*_hotspot_min_deviation` | Min channel − mean of all channels |
| `*_hotspot_range` | Max channel − min channel |
| `*_hotspot_cv` | Coefficient of variation across channels |
| `temp_press_corr` | Pearson correlation between temp and pressure |
| `temp_imp_corr` | Pearson correlation between temp and impedance |

---

## Models

### Isolation Forest

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Enough trees for stable anomaly scoring |
| `contamination` | 0.05 | Expect ~5% anomalous windows even in "healthy" training set |
| `max_samples` | auto | Subsample size for each tree |
| Scaler | StandardScaler | Normalizes features to zero mean, unit variance |

### XGBoost Classifier

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Number of boosting rounds |
| `max_depth` | 6 | Controls tree complexity (prevents overfitting) |
| `learning_rate` | 0.1 | Learning rate / shrinkage |
| `subsample` | 0.8 | Row subsampling per tree (regularization) |
| `colsample_bytree` | 0.8 | Feature subsampling per tree (regularization) |
| `eval_metric` | logloss | Binary cross-entropy loss |

---

## Pluggable Model Registry

### The Problem

Without a registry, adding a new algorithm means:
1. Editing `train_anomaly.py` or `train_classifier.py`
2. Adding import statements
3. Adding if-else branches
4. Duplicating hyperparameter handling
5. Potentially breaking existing code

### The Solution

The **model registry** (`ml/pipelines/model_registry.py`) is a centralized catalog where every algorithm is registered once with:
- **Builder function** — creates a model instance from parameters
- **Default parameters** — sensible defaults so it works out of the box
- **Optuna search space** — defines how Optuna should explore hyperparameters

**To add a new algorithm to the pipeline:**

```python
# In model_registry.py, add inside _register_builtins():

def _build_my_model(**kw):
    from my_library import MyModel
    return MyModel(**kw)

def _my_model_space(trial):
    return {
        "param_a": trial.suggest_int("param_a", 10, 100),
        "param_b": trial.suggest_float("param_b", 0.01, 1.0),
    }

register(ModelSpec(
    name="MyModel",
    category="classifier",       # or "anomaly"
    builder=_build_my_model,
    default_params={"param_a": 50, "param_b": 0.1},
    search_space=_my_model_space,
))
```

Then use it in config:
```yaml
classifier:
  algorithm: "MyModel"
  params:
    param_a: 50
```

**That's it.** No changes to training code, evaluation, or pipeline runner.

### Currently Registered Models

| Name | Category | Library | Notes |
|---|---|---|---|
| `IsolationForest` | anomaly | sklearn | Default anomaly detector |
| `OneClassSVM` | anomaly | sklearn | Boundary-based anomaly detection |
| `LocalOutlierFactor` | anomaly | sklearn | Density-based anomaly detection |
| `Autoencoder` | anomaly | PyTorch | Reconstruction-error based (DL) |
| `XGBClassifier` | classifier | xgboost | Default classifier, falls back to RF |
| `RandomForestClassifier` | classifier | sklearn | Ensemble of decision trees |
| `GradientBoostingClassifier` | classifier | sklearn | Sequential boosting |
| `SVC` | classifier | sklearn | Support vector machine |
| `MLPClassifier` | classifier | PyTorch | Multi-layer perceptron (DL) |

### Deep Learning Support

DL models live in `ml/pipelines/dl_models.py` and implement an sklearn-compatible interface (`fit`, `predict`, `predict_proba`, `score_samples`). This means:
- They work with the same training code as ML models
- They integrate with Optuna tuning automatically
- They integrate with SHAP explanations (via KernelExplainer)
- They integrate with MLflow model logging

PyTorch is an **optional dependency** — if not installed, DL models simply raise a helpful error when you try to use them. ML models work fine without it.

**Analogy:** The model registry is like an app store. Each algorithm is an "app" you can install. The pipeline is the phone — it knows how to run any app you plug in. Adding a new algorithm is like adding a new app to the store, not rebuilding the phone.

---

## Hyperparameter Tuning (Optuna)

### What is Hyperparameter Tuning?

Every ML algorithm has knobs to turn — number of trees, learning rate, depth, regularization. These are "hyperparameters" that aren't learned from data; they're set before training. Finding the best combination is **hyperparameter tuning**.

### Why Optuna?

Optuna uses **Bayesian optimization** — it learns from previous trials to suggest smarter parameter combinations. Unlike grid search (try every combination) or random search (try random ones), Optuna:
- Explores promising regions more
- Prunes bad trials early (stops wasting time on clearly poor settings)
- Scales to high-dimensional search spaces

### How it Works in Our Pipeline

```
┌─────────────────────┐
│   ml_config.yaml     │  n_trials: 50, timeout: 300s
│   optuna section     │  metric: f1_weighted
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   tuning.py          │  For each trial:
│                      │  1. Get search space from model registry
│                      │  2. Suggest params via Optuna
│                      │  3. Train model with suggested params
│                      │  4. Evaluate on validation set
│                      │  5. Report score back to Optuna
│                      │  Repeat for n_trials...
└──────────┬──────────┘
           │
           ▼
    best_params → passed to train_anomaly / train_classifier
```

### Usage

```bash
# Run pipeline with tuning
python -m ml.run_pipeline --tune

# Tune + track results in MLflow
python -m ml.run_pipeline --tune --track
```

### Configuration

```yaml
optuna:
  n_trials: 50                     # number of trials
  timeout: 300                     # max seconds
  classifier_metric: "f1_weighted" # what to optimize
  classifier_direction: "maximize"
  anomaly_direction: "maximize"
```

**Analogy:** Tuning is like adjusting a recipe. Grid search tries every possible combination of salt, pepper, and sugar — tedious. Random search picks random amounts — hit or miss. Optuna is like a chef who tastes each attempt and says "a bit more salt, less sugar" — converging on the best recipe efficiently.

---

## Experiment Tracking (MLflow)

### What is Experiment Tracking?

When running ML experiments, you want to know:
- What parameters did I use?
- What metrics did I get?
- Which run was the best?
- Can I reproduce it?

MLflow answers all of these by logging everything automatically.

### How it Works in Our Pipeline

The `ExperimentTracker` (`ml/pipelines/tracking.py`) wraps MLflow with graceful degradation — if MLflow isn't installed or tracking is disabled, all calls silently become no-ops. Zero code changes needed.

```
┌─────────────────────┐
│   run_pipeline.py    │
│                      │
│   with tracker.start_run("anomaly_IF"):
│       tracker.log_params(...)         → MLflow logs parameters
│       train model...
│       tracker.log_metrics(...)        → MLflow logs AUC, F1, etc.
│       tracker.log_artifact(...)       → MLflow saves plots, models
│                      │
└──────────┬──────────┘
           │
           ▼
    mlruns/                            ← Local MLflow tracking directory
    └── experiment_id/
        └── run_id/
            ├── params/                ← All hyperparameters
            ├── metrics/               ← AUC-ROC, F1, precision, recall
            └── artifacts/             ← Plots, models, JSON metrics
```

### Usage

```bash
# Run pipeline with MLflow tracking
python -m ml.run_pipeline --track

# View results in MLflow UI
mlflow ui
# Then open http://localhost:5000
```

### Configuration

```yaml
mlflow:
  enabled: false                # or use --track flag
  tracking_uri: "mlruns"       # local dir or remote server
  experiment_name: "bra_pipeline"
```

### What Gets Logged

| Category | What's Logged | Example |
|---|---|---|
| Parameters | Algorithm name, sample counts, feature counts, tuned params | `anomaly_algorithm: IsolationForest` |
| Metrics | AUC-ROC, F1, precision, recall, accuracy, AUC-PR | `f1: 0.92` |
| Artifacts | ROC curves, confusion matrices, SHAP plots, model files | `anomaly_roc_curve.png` |
| Tags | Run name, mode, pipeline version | `run_name: anomaly_IsolationForest` |

**Analogy:** MLflow is like a lab notebook for data science. Every experiment gets a dated entry with what you tried, what you measured, and what you got. You can flip back through the notebook to compare, reproduce, or pick the best result.

---

## Evaluation & Metrics

### Why These Metrics?

| Metric | What it tells you | Medical relevance |
|---|---|---|
| **AUC-ROC** | Overall discriminative ability | "Can the model tell healthy from anomaly?" |
| **AUC-PR** | Performance on the minority class | Better than AUC-ROC when data is imbalanced |
| **Recall** | Fraction of anomalies caught | The most critical metric — we must not miss anomalies |
| **Precision** | Fraction of flagged items that are truly anomalous | How many false alarms do we generate? |
| **F1** | Balance of precision and recall | Single summary number |

### Output Artifacts

All saved to `ml/models/simulation/evaluation/`:

- `anomaly_metrics.json` / `classifier_metrics.json` — Full metrics breakdown
- `*_roc_curve.png` — ROC curves with AUC
- `*_pr_curve.png` — Precision-Recall curves with AUC
- `*_confusion_matrix.png` — Annotated confusion matrices
- `*_score_distribution.png` — Score histograms separated by class

---

## Explainability (SHAP)

### What SHAP Does

SHAP assigns each feature a "contribution score" for each prediction. For a given window predicted as anomalous, SHAP tells you:

- "The temperature asymmetry (left–right diff) pushed the score UP by 0.15"
- "The impedance slope pushed the score UP by 0.08"
- "The ambient humidity pushed the score DOWN by 0.01"

This is based on Shapley values from cooperative game theory — each feature is treated as a "player" and its contribution is computed as the average marginal effect across all possible feature subsets.

### Output Artifacts

- `*_shap_summary.png` — Beeswarm plot: each dot is one feature for one window, colored by feature value, positioned by SHAP value
- `*_shap_bar.png` — Top-20 features by mean absolute SHAP value
- `*_feature_importance.json` — Full ranked importance for programmatic use

**Analogy:** SHAP is like an itemized receipt. The total bill is the model's prediction. Instead of just seeing "Total: anomaly," you see every line item that contributed: temperature +$0.15, pressure +$0.08, humidity −$0.01.

---

## Configuration Reference

All configuration lives in `ml/config/ml_config.yaml`. Key settings:

```yaml
windowing:
  window_s: 30.0      # Window duration (seconds)
  step_s: 5.0         # Sliding step (seconds)
  hz: 2.0             # Sampling rate

split:
  test_size: 0.2      # 20% held out for testing
  random_state: 42    # Reproducibility

anomaly_model:
  algorithm: "IsolationForest"   # any registered anomaly model
  params:
    n_estimators: 200
    contamination: 0.05

classifier:
  algorithm: "XGBClassifier"     # any registered classifier
  params:
    n_estimators: 200
    max_depth: 6

evaluation:
  score_threshold: 0.5   # Binary decision threshold

explain:
  max_samples: 200       # SHAP subsample for speed
  top_n_features: 20     # Features shown in plots

optuna:
  n_trials: 50           # Optuna trials (used with --tune)
  timeout: 300           # Max seconds for tuning
  classifier_metric: "f1_weighted"

mlflow:
  enabled: false         # Enable with --track flag
  tracking_uri: "mlruns"
  experiment_name: "bra_pipeline"
```
    max_depth: 6

evaluation:
  score_threshold: 0.5   # Binary decision threshold

explain:
  max_samples: 200       # SHAP subsample for speed
  top_n_features: 20     # Features shown in plots
```

---

## How to Run

### Prerequisites

```bash
# From the project root, with venv activated:
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python -m ml.run_pipeline
```

### Run Only Anomaly Detector

```bash
python -m ml.run_pipeline --mode anomaly
```

### Run Only Classifier

```bash
python -m ml.run_pipeline --mode classifier
```

### Skip SHAP (Faster)

```bash
python -m ml.run_pipeline --no-explain
```

### With Optuna Hyperparameter Tuning

```bash
python -m ml.run_pipeline --tune
```

### With MLflow Experiment Tracking

```bash
python -m ml.run_pipeline --track
```

### All Bells and Whistles

```bash
python -m ml.run_pipeline --tune --track
```

### View MLflow Dashboard

```bash
mlflow ui
# Open http://localhost:5000
```

### Swap Algorithm (Config Change Only)

```yaml
# In ml/config/ml_config.yaml, change:
anomaly_model:
  algorithm: "OneClassSVM"      # was "IsolationForest"
classifier:
  algorithm: "RandomForestClassifier"  # was "XGBClassifier"
# Then re-run: python -m ml.run_pipeline
```

### Enable Deep Learning Models

```bash
# Install PyTorch first:
pip install torch

# Then configure in ml_config.yaml:
# anomaly_model:
#   algorithm: "Autoencoder"
# classifier:
#   algorithm: "MLPClassifier"

python -m ml.run_pipeline
```

---

## Folder Structure

```
ml/
├── __init__.py
├── run_pipeline.py                  ← CLI entry point (--tune, --track, --mode, --no-explain)
├── config/
│   └── ml_config.yaml               ← All hyperparameters, tuning, and tracking settings
├── data/
│   └── raw/
│       ├── simulated/
│       │   ├── healthy/              ← 500 healthy CSVs (from Phase 1)
│       │   └── anomaly/              ← 150 anomaly CSVs (from Phase 1)
│       └── hardware/
│           ├── clinical/
│           └── healthy/
├── models/
│   └── simulation/                   ← Created by pipeline
│       ├── anomaly_model.joblib
│       ├── scaler.joblib
│       ├── feature_names.json
│       ├── classifier_model.joblib
│       ├── classifier_scaler.joblib
│       ├── classifier_feature_names.json
│       └── evaluation/
│           ├── anomaly_metrics.json
│           ├── classifier_metrics.json
│           ├── *_roc_curve.png
│           ├── *_pr_curve.png
│           ├── *_confusion_matrix.png
│           ├── *_score_distribution.png
│           ├── *_shap_summary.png
│           ├── *_shap_bar.png
│           └── *_feature_importance.json
├── pipelines/
│   ├── preprocess.py                 ← Phase 1: load, validate, clean CSVs
│   ├── feature_engineering.py        ← Phase 2: extract ~389 features/window
│   ├── model_registry.py             ← Phase 2: pluggable algorithm factory
│   ├── dl_models.py                  ← Phase 2: PyTorch DL model wrappers
│   ├── train_anomaly.py              ← Phase 2: anomaly model training (registry-based)
│   ├── train_classifier.py           ← Phase 2: classifier training (registry-based)
│   ├── tuning.py                     ← Phase 2: Optuna hyperparameter tuning
│   ├── tracking.py                   ← Phase 2: MLflow experiment tracking
│   ├── evaluate.py                   ← Phase 2: metrics + plots
│   └── explain.py                    ← Phase 2: SHAP explainability
└── utils/
    ├── __init__.py
    ├── features.py                   ← Low-level feature functions
    ├── visualization.py              ← Reusable plot helpers
    └── windowing.py                  ← Sliding window utilities
```

---

## Summary

Phase 2 transforms raw sensor CSV data into actionable anomaly predictions with full explainability, backed by a modular, production-grade ML infrastructure:

1. **Preprocessing** (Phase 1) — Cleans and validates raw data
2. **Windowing** — Slices 5-minute sessions into 30-second analysis windows
3. **Feature Engineering** — Extracts ~389 statistical, spectral, and spatial features per window
4. **Model Registry** — Pluggable architecture supporting any ML or DL algorithm via config
5. **Hyperparameter Tuning** — Optuna Bayesian optimization finds optimal model settings
6. **Training** — Two complementary model categories (unsupervised + supervised), any algorithm
7. **Evaluation** — Comprehensive metrics and diagnostic plots
8. **Explainability** — SHAP values reveal *why* each prediction was made
9. **Experiment Tracking** — MLflow logs params, metrics, artifacts for reproducibility

This pipeline is fully automated via `python -m ml.run_pipeline` and supports:
- `--tune` for Optuna hyperparameter tuning
- `--track` for MLflow experiment tracking  
- Swapping algorithms by changing one line in `ml/config/ml_config.yaml`
