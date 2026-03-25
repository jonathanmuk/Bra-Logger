# BRA-Logger: AI-Powered Breast Anomaly Risk Scoring System

## Complete Implementation Plan

> **Project Goal**: Develop an IoT-based wearable smart bra system that uses embedded temperature, pressure, and bioimpedance sensors to capture breast tissue data, process it through an AI/ML pipeline, and generate anomaly risk scores — assisting in the early identification of abnormal breast tissue patterns.

---

## Table of Contents

1. [Problem Statement & Clinical Rationale](#1-problem-statement--clinical-rationale)
2. [Data Strategy — Datasets & Collection](#2-data-strategy--datasets--collection)
3. [Hardware Requirements — Complete Bill of Materials](#3-hardware-requirements--complete-bill-of-materials)
4. [Tech Stack](#4-tech-stack)
5. [Project Structure — Folder Architecture](#5-project-structure--folder-architecture)
6. [Software Implementation — Phased Breakdown](#6-software-implementation--phased-breakdown)
7. [ML/AI Pipeline Design](#7-mlai-pipeline-design)
8. [API Design](#8-api-design)
9. [Frontend Dashboard Design](#9-frontend-dashboard-design)
10. [Deployment & DevOps](#10-deployment--devops)
11. [Testing Strategy](#11-testing-strategy)
12. [Ethical, Regulatory & Legal Considerations](#12-ethical-regulatory--legal-considerations)

---

## 1. Problem Statement & Clinical Rationale

### Why Temperature & Pressure?

Cancerous and pre-cancerous breast tissue exhibits measurable physiological differences:

| Biomarker | Normal Tissue | Abnormal/Cancerous Tissue | Clinical Basis |
|---|---|---|---|
| **Surface Temperature** | Symmetric across both breasts (±0.3°C) | Localized elevated temperature (0.5–2°C higher) due to increased angiogenesis and metabolic activity | Breast thermography literature (Gautherie, 1980; Kontos et al., 2011) |
| **Tissue Stiffness (Pressure)** | Soft, uniform resistance under compression | Hard, non-uniform lumps with higher stiffness | Clinical breast exam (CBE) principles — tumors are typically 5–10x stiffer than surrounding tissue |
| **Bioelectrical Impedance** | Higher impedance (~600–1000 Ω at 50 kHz) | Lower impedance (~200–500 Ω) due to increased vascularity and cellular disruption | Bioimpedance spectroscopy (BIS) research (Zou & Guo, 2003) |
| **Temperature Asymmetry (L vs R)** | Mean ΔT < 0.5°C | Mean ΔT > 1.0°C sustained over time | Key criterion in interpreting thermographic images |

### What We're Detecting

We **do not** diagnose cancer. We compute an **anomaly risk score** (0–100) based on statistical deviations from a baseline "healthy" pattern across multiple modalities. This score is a **screening aid**, not a diagnostic tool.

---

## 2. Data Strategy — Datasets & Collection

### 2.1 The Dataset Problem

We do **not** have a pre-built dataset of temperature + pressure + bioimpedance sensor readings from wearable devices. This is novel research, and such datasets don't exist publicly. We have two parallel strategies:

### 2.2 Strategy A — Synthetic + Simulation Data (Start Here)

This is what the existing `receiver_logger.py` already does. We expand it:

| Parameter | Value | Rationale |
|---|---|---|
| **Healthy baseline samples** | 500–1,000 sessions × 5 min each | Captures normal variation across time-of-day, activity level, menstrual cycle phase |
| **Simulated anomaly samples** | 100–200 sessions | Injected hotspots, asymmetries, stiffness zones |
| **Sampling rate** | 2 Hz (existing) | 2 samples/second is sufficient for slow-changing physiological signals |
| **Channels** | 4 temp + 8 pressure + 4 impedance = **16 channels** | Extended from current 12 to include bioimpedance |
| **Minimum dataset size for MVP** | ~50,000 windows (30s sliding window, 5s step) | Enough for Isolation Forest or autoencoder training |

**How to simulate anomalies:**
- **Temperature hotspot**: Add +0.5 to +2.0°C to 1–2 specific temp sensors for a sustained period
- **Pressure lump**: Increase 1–2 pressure channels by 30–60% (simulating a hard mass)
- **Impedance drop**: Reduce impedance on 1–2 channels by 40–60%
- **Asymmetry**: Create a persistent L-R difference exceeding 1.0°C or 25% pressure differential

### 2.3 Strategy B — Public Datasets for Transfer Learning

These cannot be used directly (different modality), but are valuable for validating feature engineering and model architectures:

| Dataset | Source | Content | Use Case |
|---|---|---|---|
| **DMR-IR (Database for Mastology Research)** | [Visual Lab UFPE](http://visual.ic.uff.br/dmi) | 293 patients, IR images + mammograms + clinical data | Validate thermal pattern features; benchmark classification |
| **Mendeley Breast Thermography** | [Mendeley Data](https://data.mendeley.com) | 119 women, FLIR A300 thermal images (benign/malignant labels) | Train thermal image classifiers; extract statistical thermal features |
| **Kaggle Thermal Breast** | [Kaggle](https://www.kaggle.com/datasets) | Thermography images categorized normal/sick/unknown | Quick prototyping of deep learning classifiers |
| **UCI IR Temperature** | [UCI ML Repository](https://archive.ics.uci.edu) | Temperature readings + oral temps + demographics | Regression baseline; understand normal temperature variance |

### 2.4 Strategy C — Real-World Data Collection Protocol (Post-Hardware)

Once the wearable is built:

1. **Healthy Baseline Collection** (Phase 1 priority):
   - Recruit 20–50 healthy volunteers (IRB approval required for any research publication)
   - Each wears the device for 5 min in 3 conditions: rest, standing, post-light-walk
   - Collect 3 sessions per participant across different times of day
   - Label: `healthy`, with metadata: age, BMI, menstrual phase, ambient temperature

2. **Known-Anomaly Collection** (Phase 2, if clinical access available):
   - Partner with a clinic or hospital
   - Collect readings from patients with known benign lumps or confirmed malignancies
   - Label: `benign`, `malignant`, `post-surgery`

3. **Minimum Viable Dataset**:

| Category | Sessions | Per-Session Duration | Total Windows (~30s, 5s step) |
|---|---|---|---|
| Healthy | 150 | 5 min | ~7,500 |
| Simulated Anomaly | 100 | 5 min | ~5,000 |
| Real Anomaly (stretch) | 20+ | 5 min | ~1,000 |
| **Total MVP** | **270+** | | **~13,500+** |

> **Recommendation**: Start with Strategy A (simulation). Use Strategy B for model architecture validation. Move to Strategy C once hardware is functional and tested.

---

## 3. Hardware Requirements — Complete Bill of Materials

### 3.1 Core Controller

| Component | Specification | Quantity | Purpose | Est. Cost (USD) |
|---|---|---|---|---|
| **ESP32-WROOM-32D** DevKit | Dual-core 240 MHz, Wi-Fi + BLE, 4 MB flash | 1 | Main MCU — sensor sampling, Wi-Fi AP, data streaming | $5–8 |
| USB-C cable | For programming & power | 1 | Flashing firmware, serial debug | $3 |

### 3.2 Temperature Sensors

| Component | Specification | Quantity | Purpose | Est. Cost |
|---|---|---|---|---|
| **DS18B20** (waterproof, stainless steel probe) | Digital, ±0.5°C accuracy, 1-Wire protocol | 4 | 2 per breast — measure surface temperature at quadrant positions | $4–6 (pack of 5) |
| **4.7 kΩ resistor** | Pull-up resistor for 1-Wire data line | 1 | Required by DS18B20 1-Wire protocol | $0.10 |

> **Sensor placement**: Upper-outer and lower-inner quadrant of each breast (the upper-outer quadrant is where ~50% of breast cancers originate).

**Why DS18B20?**
- Digital output = no ADC needed, less noise
- ±0.5°C is adequate for detecting 1–2°C anomalies
- Multiple sensors on a single GPIO pin (1-Wire bus)
- Waterproof probes are skin-safe

**Upgrade path**: For higher accuracy (±0.25°C), consider **ADT7420** (I2C digital sensor) at ~$5 each.

### 3.3 Pressure/Force Sensors

| Component | Specification | Quantity | Purpose | Est. Cost |
|---|---|---|---|---|
| **FSR 402** (Force Sensing Resistor) | 0–10 kg, 12.7mm active area | 8 | 4 per breast — detect tissue stiffness under gentle compression | $16–24 (pack of 8) |
| **10 kΩ resistors** | Voltage divider for each FSR | 8 | Convert FSR resistance change to voltage for ADC reading | $1 |
| **CD74HC4067** 16-channel analog MUX | Breakout board | 1 | Multiplex 8+ analog sensors into ESP32's single ADC | $2–3 |

> **Sensor placement**: 4 per breast, in a cross/diamond pattern at the 12, 3, 6, 9 o'clock positions of each breast cup.

**Why FSR 402?**
- Thin, flexible, embeddable in fabric
- Simple analog output (voltage divider → ADC)
- Sufficient sensitivity for detecting tissue stiffness differences
- Clinically, manual palpation uses similar principles

### 3.4 Bioimpedance Sensors (Phase 2 Addition)

| Component | Specification | Quantity | Purpose | Est. Cost |
|---|---|---|---|---|
| **AD5933** impedance converter | Breakout board (I2C), 1 kHz–100 kHz sweep | 1 | Measure tissue electrical impedance at multiple frequencies | $15–20 |
| **Ag/AgCl gel electrodes** (disposable) | Pre-gelled, snap-on, medical grade | 8 (4 pairs) | Skin-contact electrode pairs for 4-point impedance measurement | $5 (pack of 50) |
| **Electrode snap connectors** | Standard 3.5mm or 4mm snap | 4 | Connect electrodes to AD5933 board | $3 |

> **Why bioimpedance?** Cancerous tissue has measurably lower impedance due to increased vascularity and cell membrane disruption. Adding this modality increases specificity beyond temperature and pressure alone.

### 3.5 Additional Sensors (Optional but Recommended)

| Component | Specification | Quantity | Purpose | Est. Cost |
|---|---|---|---|---|
| **DHT22** / **BME280** | Ambient temp + humidity sensor | 1 | Environmental correction — body surface temp is affected by ambient conditions | $3–5 |
| **MPU6050** / **MPU9250** IMU | Accelerometer + gyroscope (I2C) | 1 | Detect motion artifacts, patient posture, filter noise | $3–5 |

### 3.6 Power

| Component | Specification | Quantity | Purpose | Est. Cost |
|---|---|---|---|---|
| **3.7V LiPo battery** | 1000–2000 mAh, JST connector | 1 | Portable power for wearable operation | $5–8 |
| **TP4056** micro-USB charging module | Li-ion charger with protection circuit | 1 | Safe LiPo charging | $1–2 |
| **Slide switch** (SPST) | Mini toggle or slide | 1 | Hard power on/off | $0.50 |

### 3.7 Wiring & Assembly

| Component | Quantity | Purpose | Est. Cost |
|---|---|---|---|
| Breadboard (for prototyping) | 1 | Initial circuit prototyping | $3 |
| Jumper wires (M-M, M-F, F-F assorted) | 40-pack | Sensor connections during prototyping | $3 |
| Perfboard / Stripboard | 1 | Soldered prototype (post-breadboard) | $2 |
| Header pins (male + female) | 2 strips | Modular connections on perfboard | $1 |
| Heat shrink tubing assortment | 1 pack | Insulate solder joints | $2 |
| Soldering iron + solder wire | 1 set | Assembly (if not already owned) | $15–25 |
| Hot glue gun | 1 | Sensor mounting on bra fabric | $5 |
| Flexible ribbon cable / thin gauge wire | 1m | Route sensor wires inside bra fabric channels | $2 |
| 3D-printed enclosure (optional) | 1 | House ESP32 + battery + MUX neatly | $5–10 (filament) |
| Sports bra (for embedding) | 1 | Physical housing for sensors | $10–15 |

### 3.8 Total Estimated Hardware Cost

| Category | Cost Range |
|---|---|
| Core electronics | $30–50 |
| Sensors (temp + pressure + impedance) | $40–55 |
| Power system | $7–11 |
| Wiring & assembly materials | $25–45 |
| **Total Prototype** | **$100–160** |

### 3.9 Hardware-to-Software Communication Architecture

```
┌─────────────────────────────────────────────────┐
│                  SMART BRA                      │
│                                                 │
│  DS18B20 ──(1-Wire)──┐                         │
│  FSR 402 ──(Analog)──┤                         │
│  AD5933 ──(I2C)──────┤   ┌─────────────┐       │
│  BME280 ──(I2C)──────┼──►│  ESP32       │       │
│  MPU6050 ─(I2C)──────┤   │  (2 Hz poll) │       │
│  CD74HC4067 ─────────┘   └──────┬──────┘       │
│                                  │              │
└──────────────────────────────────┼──────────────┘
                                   │
                            Wi-Fi AP (HTTP)
                            or BLE Serial
                                   │
                    ┌──────────────▼──────────────┐
                    │     RECEIVER (Laptop/RPi)    │
                    │   receiver_logger.py         │
                    │   Polls GET /data every 0.5s │
                    │   Writes CSV files           │
                    └──────────────┬───────────────┘
                                   │
                            CSV files / REST API
                                   │
                    ┌──────────────▼──────────────┐
                    │     BACKEND (FastAPI)        │
                    │   Ingests data, runs ML,     │
                    │   stores results, serves API │
                    └──────────────┬───────────────┘
                                   │
                              REST + WebSocket
                                   │
                    ┌──────────────▼──────────────┐
                    │     FRONTEND (React)         │
                    │   Dashboard: heatmaps,       │
                    │   charts, risk scores, alerts│
                    └─────────────────────────────┘
```

---

## 4. Tech Stack

### 4.1 Firmware (Embedded)

| Technology | Purpose |
|---|---|
| **PlatformIO + Arduino framework** | Build system and HAL for ESP32 |
| **C/C++** | Firmware language |
| **OneWire + DallasTemperature** libraries | DS18B20 driver |
| **Wire (I2C)** | AD5933, BME280, MPU6050 communication |
| **WiFi + WebServer** (ESP32) | HTTP API to stream sensor data |
| **ArduinoJson** | Structured JSON responses |

### 4.2 Data Collection & Processing

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language for data scripts, ML, backend |
| **Pandas** | Data loading, cleaning, windowing |
| **NumPy** | Numerical computation, feature engineering |
| **SciPy** | Signal processing (filtering, FFT) |

### 4.3 Machine Learning

| Technology | Purpose |
|---|---|
| **Scikit-learn** | Isolation Forest, Random Forest, StandardScaler, evaluation metrics |
| **XGBoost** | Gradient-boosted classification/regression for risk scoring |
| **PyTorch** (optional Phase 3) | Autoencoder for unsupervised anomaly detection |
| **Joblib** | Model serialization/deserialization |
| **Optuna** (optional) | Hyperparameter optimization |
| **SHAP** | Model explainability — which sensors/features drive the risk score |

### 4.4 Backend

| Technology | Purpose |
|---|---|
| **FastAPI** | REST API framework (async, high-performance, auto-docs) |
| **Uvicorn** | ASGI server |
| **SQLAlchemy** | ORM for database models |
| **Alembic** | Database migrations |
| **PostgreSQL** | Production database (structured patient/session data) |
| **SQLite** | Development/testing database |
| **Pydantic** | Request/response validation |
| **python-jose + passlib** | JWT authentication |
| **WebSocket** (FastAPI) | Real-time streaming of live sensor data |

### 4.5 Frontend

| Technology | Purpose |
|---|---|
| **React 18+** (Vite) | SPA dashboard framework |
| **TypeScript** | Type-safe frontend development |
| **Recharts** / **Chart.js** | Time-series charts for sensor data |
| **React-Heatmap-Grid** or **D3.js** | Breast heatmap visualization |
| **Axios** | HTTP client for API calls |
| **TanStack Query (React Query)** | Server state management and caching |
| **React Router** | Client-side routing |
| **Zustand** or **Context API** | Client state management |
| **Tailwind CSS** or **Material UI** | UI styling (user preference) |

### 4.6 DevOps & Tools

| Technology | Purpose |
|---|---|
| **Git / GitHub** | Version control |
| **Docker + Docker Compose** | Containerized development & deployment |
| **pytest** | Backend testing |
| **Vitest / Jest** | Frontend testing |
| **pre-commit** | Linting hooks (ruff, black, eslint) |
| **GitHub Actions** | CI/CD pipeline |

---

## 5. Project Structure — Folder Architecture

```
bra_logger/
│
├── README.md                          # Project overview and setup guide
├── implementation.md                  # This file
├── .gitignore
├── .env.example                       # Environment variable template
├── docker-compose.yml                 # Multi-service orchestration
│
├── firmware/                          # ══════ EMBEDDED (ESP32) ══════
│   ├── platformio.ini                 # PlatformIO configuration
│   ├── src/
│   │   ├── main.cpp                   # Entry point — setup/loop
│   │   ├── config.h                   # Pin definitions, Wi-Fi creds, sampling rate
│   │   ├── sensors/
│   │   │   ├── temperature.h/.cpp     # DS18B20 driver (1-Wire)
│   │   │   ├── pressure.h/.cpp        # FSR via CD74HC4067 MUX
│   │   │   ├── impedance.h/.cpp       # AD5933 bioimpedance driver
│   │   │   └── environment.h/.cpp     # BME280 ambient temp/humidity
│   │   ├── communication/
│   │   │   ├── wifi_ap.h/.cpp         # Wi-Fi Access Point setup
│   │   │   ├── http_server.h/.cpp     # HTTP endpoints (/data, /health, /config)
│   │   │   └── ble_serial.h/.cpp      # Optional BLE fallback
│   │   └── utils/
│   │       ├── led_status.h/.cpp      # Status LED patterns
│   │       └── power_mgmt.h/.cpp      # Sleep modes, battery monitor
│   ├── include/                       # Shared headers
│   ├── lib/                           # Local libraries
│   └── test/                          # Firmware unit tests (Unity framework)
│
├── data_collection/                   # ══════ DATA INGESTION ══════
│   ├── receiver_logger.py             # Poll ESP32 HTTP → CSV (refactored)
│   ├── serial_logger.py               # USB serial fallback logger
│   ├── simulate_data.py               # Advanced simulation with anomaly injection
│   ├── config.yaml                    # Logger configuration (URL, Hz, output path)
│   └── sessions/                      # Raw CSV files organized by session
│       ├── healthy/
│       ├── simulated_anomaly/
│       └── clinical/                  # Future: real patient data
│
├── ml/                                # ══════ MACHINE LEARNING ══════
│   ├── config/
│   │   └── ml_config.yaml             # Hyperparameters, feature sets, model selection
│   ├── data/
│   │   ├── raw/
│   │   │   ├── simulated/             # Synthetic data (no hardware needed)
│   │   │   │   ├── healthy/           # Normal baseline sessions
│   │   │   │   ├── anomaly/           # Injected anomaly sessions
│   │   │   │   └── manifest.json      # Labels + generation parameters
│   │   │   └── hardware/              # Real sensor data (post-hardware)
│   │   │       ├── healthy/
│   │   │       ├── clinical/
│   │   │       └── manifest.json
│   │   ├── processed/
│   │   │   ├── simulated/             # Feature matrices from sim data
│   │   │   │   └── splits/            # train/val/test
│   │   │   └── hardware/              # Feature matrices from real data
│   │   │       └── splits/
│   │   └── registry.json              # Dataset version tracking
│   ├── pipelines/                     # ★ SHARED pipeline code (sim + prod use same code)
│   │   ├── preprocess.py              # Clean → validate → impute → normalize
│   │   ├── feature_engineering.py     # Window features, spatial features, spectral
│   │   ├── train_anomaly.py           # Unsupervised: Isolation Forest / Autoencoder
│   │   ├── train_classifier.py        # Supervised: XGBoost / Random Forest
│   │   ├── evaluate.py                # Metrics, confusion matrix, ROC/PR curves
│   │   └── explain.py                 # SHAP values, feature importance plots
│   ├── models/
│   │   ├── simulation/                # Model trained on simulated data
│   │   │   ├── anomaly_model.joblib
│   │   │   ├── scaler.joblib
│   │   │   ├── feature_columns.json
│   │   │   ├── training_metadata.json
│   │   │   └── evaluation/            # metrics.json + plots/
│   │   ├── production/                # Model trained on real data (later)
│   │   │   └── (same structure)
│   │   └── active/ → simulation/      # Symlink: backend loads from here
│   ├── notebooks/
│   │   ├── 01_train_simulation_model.ipynb
│   │   ├── 02_train_production_model.ipynb
│   │   ├── 03_eda_simulated.ipynb
│   │   ├── 04_eda_hardware.ipynb
│   │   ├── 05_feature_exploration.ipynb
│   │   └── 06_model_comparison.ipynb
│   ├── utils/
│   │   ├── windowing.py               # Sliding window utilities
│   │   ├── features.py                # Statistical feature extraction functions
│   │   └── visualization.py           # Plotting helpers
│   ├── tests/
│   │   ├── test_preprocess.py
│   │   ├── test_features.py
│   │   └── test_model.py
│   └── run_pipeline.py                # CLI: train/predict --source simulated|hardware
│
├── backend/                           # ══════ FASTAPI BACKEND ══════
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app factory, middleware, lifespan
│   │   ├── config.py                  # Settings (Pydantic BaseSettings from .env)
│   │   ├── database.py                # SQLAlchemy engine, session factory
│   │   ├── models/                    # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── patient.py             # Patient demographics
│   │   │   ├── session.py             # Scan session metadata
│   │   │   ├── reading.py             # Raw sensor readings
│   │   │   └── risk_score.py          # Computed risk scores per session/window
│   │   ├── schemas/                   # Pydantic request/response schemas
│   │   │   ├── __init__.py
│   │   │   ├── patient.py
│   │   │   ├── session.py
│   │   │   ├── reading.py
│   │   │   └── risk_score.py
│   │   ├── api/                       # Route handlers grouped by resource
│   │   │   ├── __init__.py
│   │   │   ├── router.py              # Main API router
│   │   │   ├── patients.py            # CRUD: /api/patients
│   │   │   ├── sessions.py            # CRUD: /api/sessions
│   │   │   ├── readings.py            # POST raw data, GET historical
│   │   │   ├── analysis.py            # POST /api/analyze, GET /api/risk-scores
│   │   │   ├── live.py                # WebSocket: /ws/live — real-time sensor feed
│   │   │   └── auth.py                # Login, register, token refresh
│   │   ├── services/                  # Business logic layer
│   │   │   ├── __init__.py
│   │   │   ├── ml_service.py          # Load model, run inference, compute risk score
│   │   │   ├── data_service.py        # Data ingestion, validation, storage
│   │   │   ├── session_service.py     # Session management logic
│   │   │   └── alert_service.py       # Threshold-based alerting
│   │   ├── core/                      # Cross-cutting concerns
│   │   │   ├── __init__.py
│   │   │   ├── security.py            # JWT, password hashing
│   │   │   ├── dependencies.py        # FastAPI Depends() factories
│   │   │   └── exceptions.py          # Custom exception handlers
│   │   └── ml/                        # ML model loading & inference bridge
│   │       ├── __init__.py
│   │       ├── model_loader.py        # Load joblib models at startup
│   │       └── predictor.py           # Feature extraction + model.predict
│   ├── alembic/                       # Database migrations
│   │   ├── env.py
│   │   └── versions/
│   ├── alembic.ini
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       ├── conftest.py
│       ├── test_api_patients.py
│       ├── test_api_sessions.py
│       ├── test_ml_service.py
│       └── test_data_service.py
│
├── frontend/                          # ══════ REACT FRONTEND ══════
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── Dockerfile
│   ├── index.html
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── main.tsx                   # React entry
│       ├── App.tsx                    # Root component + router
│       ├── api/                       # API client layer
│       │   ├── client.ts              # Axios instance, interceptors
│       │   ├── patients.ts            # Patient API calls
│       │   ├── sessions.ts            # Session API calls
│       │   ├── readings.ts            # Reading API calls
│       │   └── analysis.ts            # Risk score / analysis API calls
│       ├── components/                # Reusable UI components
│       │   ├── Layout/
│       │   │   ├── Sidebar.tsx
│       │   │   ├── Header.tsx
│       │   │   └── MainLayout.tsx
│       │   ├── Charts/
│       │   │   ├── TemperatureChart.tsx    # Time-series temp visualization
│       │   │   ├── PressureChart.tsx       # Time-series pressure visualization
│       │   │   ├── ImpedanceChart.tsx      # Time-series impedance visualization
│       │   │   └── RiskScoreGauge.tsx      # Circular gauge 0–100
│       │   ├── Heatmap/
│       │   │   └── BreastHeatmap.tsx       # Spatial heatmap overlay on breast diagram
│       │   ├── Alerts/
│       │   │   └── RiskAlert.tsx           # Color-coded risk banners
│       │   └── Common/
│       │       ├── LoadingSpinner.tsx
│       │       ├── ErrorBoundary.tsx
│       │       └── DataTable.tsx
│       ├── pages/                     # Route-level page components
│       │   ├── Dashboard.tsx          # Main overview: current risk, latest readings
│       │   ├── LiveMonitor.tsx        # Real-time WebSocket sensor feed
│       │   ├── SessionHistory.tsx     # Browse past sessions
│       │   ├── SessionDetail.tsx      # Deep dive into one session's data
│       │   ├── PatientProfile.tsx     # Patient info and scan history
│       │   ├── Analysis.tsx           # Run new analysis, view results
│       │   └── Settings.tsx           # Device config, thresholds, preferences
│       ├── hooks/                     # Custom React hooks
│       │   ├── useWebSocket.ts        # WebSocket connection management
│       │   ├── useSensorData.ts       # React Query hooks for sensor data
│       │   └── useRiskScore.ts        # Risk score fetching
│       ├── store/                     # State management
│       │   └── useAppStore.ts         # Zustand or Context
│       ├── types/                     # TypeScript type definitions
│       │   ├── patient.ts
│       │   ├── session.ts
│       │   ├── reading.ts
│       │   └── riskScore.ts
│       ├── utils/                     # Helper functions
│       │   ├── formatters.ts          # Date, number formatting
│       │   └── constants.ts           # API URLs, threshold values
│       └── styles/
│           ├── index.css              # Global styles
│           └── variables.css          # CSS custom properties / theme tokens
│
├── docs/                              # ══════ DOCUMENTATION ══════
│   ├── architecture.md                # System architecture overview
│   ├── hardware_setup.md              # Wiring diagrams, assembly instructions
│   ├── data_collection_protocol.md    # How to run data collection sessions
│   ├── model_documentation.md         # ML model choices, training process
│   ├── api_reference.md               # API endpoint documentation
│   └── images/                        # Diagrams, photos
│       ├── system_architecture.png
│       ├── wiring_diagram.png
│       └── sensor_placement.png
│
└── scripts/                           # ══════ UTILITY SCRIPTS ══════
    ├── setup_env.sh                   # One-command dev environment setup
    ├── seed_db.py                     # Seed database with test data
    ├── export_model.py                # Export model for deployment
    └── generate_report.py             # Generate PDF report from session data
```

---

## 6. Software Implementation — Phased Breakdown

### Phase 1: Foundation & Data Collection (Weeks 1–3)

> **Hardware required**: ESP32 + sensors for real data; simulation mode works without hardware.

| # | Task | Depends On | Output |
|---|---|---|---|
| 1.1 | Refactor firmware into modular structure (`firmware/src/sensors/`, `communication/`) | Existing `main.cpp` | Clean, maintainable firmware |
| 1.2 | Add bioimpedance sensor driver (`impedance.h/.cpp`) | AD5933 hardware | I2C read of impedance at multiple frequencies |
| 1.3 | Add ambient sensor driver (`environment.h/.cpp`) | BME280 hardware | Ambient temp/humidity in JSON payload |
| 1.4 | Expand JSON payload to include impedance + ambient + IMU channels | 1.2, 1.3 | Extended `/data` response |
| 1.5 | Refactor `receiver_logger.py` → `data_collection/receiver_logger.py` with config file | Existing script | Configurable, robust data collector |
| 1.6 | Build `simulate_data.py` — advanced simulator with anomaly injection modes | — | Synthetic data for ML training |
| 1.7 | Create data collection session management (directory structure, metadata files) | 1.5 | Organized raw data storage |
| 1.8 | Collect 500+ simulated healthy sessions + 100+ anomaly sessions | 1.6 | Training dataset |

### Phase 2: ML Pipeline (Weeks 3–5)

> **Hardware required**: No — runs on collected/simulated CSV data.

| # | Task | Depends On | Output |
|---|---|---|---|
| 2.1 | Build preprocessing pipeline (`ml/pipelines/preprocess.py`) | Phase 1 CSVs | Cleaned, validated DataFrames |
| 2.2 | Implement feature engineering (`feature_engineering.py`) — per-channel stats, L-R differences, spectral features, cross-channel correlations | 2.1 | Feature matrix (N_windows × ~389 features) |
| 2.3 | Build pluggable model registry (`model_registry.py`) — factory pattern supporting any ML/DL algorithm via config | — | Centralized algorithm catalog |
| 2.4 | Build DL model wrappers (`dl_models.py`) — PyTorch Autoencoder + MLP with sklearn interface | 2.3 | Optional deep learning support |
| 2.5 | Train anomaly detector (`train_anomaly.py`) — any registered algorithm (default: IsolationForest) | 2.2, 2.3 | `anomaly_model.joblib`, `scaler.joblib` |
| 2.6 | Train supervised classifier (`train_classifier.py`) — any registered algorithm (default: XGBClassifier) | 2.2, 2.3 | `classifier_model.joblib` |
| 2.7 | Build Optuna tuning module (`tuning.py`) — Bayesian hyperparameter optimization for any model | 2.3 | Best hyperparameters per model |
| 2.8 | Build MLflow tracking module (`tracking.py`) — experiment logging with graceful degradation | — | Experiment runs with params, metrics, artifacts |
| 2.9 | Build evaluation pipeline (`evaluate.py`) — ROC curves, precision/recall | 2.5, 2.6 | Performance metrics, visualizations |
| 2.10 | Implement SHAP explainability (`explain.py`) | 2.5 | Feature importance rankings |
| 2.11 | Create `run_pipeline.py` CLI — end-to-end with `--tune`, `--track` flags | 2.1–2.10 | Single command for train/evaluate/tune/track |
| 2.12 | EDA and model comparison notebooks | 2.2 | `notebooks/01_eda.ipynb` etc. |

#### Feature Engineering Details

The feature extraction (from `pipeline.py`, expanded) will produce ~150 features per window:

**Per-channel features (× 16 channels = ~144 features)**:
- Mean, std, min, max, range
- Linear trend (slope)
- Energy (mean of x²)
- First derivative mean & std
- Spectral features: dominant frequency, spectral entropy (via FFT)

**Spatial/Cross-channel features (~15 features)**:
- Left-Right temperature difference (mean, std, max)
- Left-Right pressure difference (mean, std, max)
- Left-Right impedance difference (mean, std, max)
- Temperature hotspot intensity (max sensor − mean of all sensors)
- Pressure hotspot intensity
- Cross-modal correlation (temp-pressure Pearson r)

### Phase 3: Backend API (Weeks 5–7)

> **Hardware required**: No — backend consumes CSV/JSON data and serves API.

| # | Task | Depends On | Output |
|---|---|---|---|
| 3.1 | Initialize FastAPI project structure (`backend/app/`) | — | Working API skeleton with health check |
| 3.2 | Define database models (Patient, Session, Reading, RiskScore) | 3.1 | SQLAlchemy models |
| 3.3 | Set up Alembic migrations | 3.2 | Migration scripts |
| 3.4 | Implement patient CRUD endpoints (`/api/patients`) | 3.2 | Create, read, update, list patients |
| 3.5 | Implement session CRUD endpoints (`/api/sessions`) | 3.2 | Create/manage scan sessions |
| 3.6 | Implement data ingestion endpoint (`POST /api/readings/bulk`) | 3.2 | Ingest raw sensor CSVs or JSON |
| 3.7 | Implement ML service (`ml_service.py`) — load model, run inference | Phase 2 models | Risk score computation |
| 3.8 | Implement analysis endpoint (`POST /api/analyze`) | 3.7 | Returns risk score + feature breakdown |
| 3.9 | Implement WebSocket endpoint (`/ws/live`) for real-time streaming | 3.1 | Live sensor data push |
| 3.10 | Add JWT authentication (`auth.py`, `security.py`) | 3.1 | Protected endpoints |
| 3.11 | Write API tests | 3.4–3.10 | Test suite |

### Phase 4: Frontend Dashboard (Weeks 7–9)

> **Hardware required**: No — frontend connects to backend API.

| # | Task | Depends On | Output |
|---|---|---|---|
| 4.1 | Initialize React + Vite + TypeScript project | — | Working React skeleton |
| 4.2 | Build layout components (Sidebar, Header, MainLayout) | 4.1 | App shell |
| 4.3 | Build Dashboard page — risk score gauge, latest readings summary | 4.2 + API | Main overview page |
| 4.4 | Build Live Monitor page — WebSocket, real-time sensor charts | 4.2 + WS | Real-time visualization |
| 4.5 | Build Session History page — table of past sessions | 4.2 + API | Browse past scans |
| 4.6 | Build Session Detail page — charts, heatmap, risk breakdown | 4.2 + API | Deep dive into session data |
| 4.7 | Build Breast Heatmap component — D3/canvas overlay on breast diagram | 4.2 | Spatial visualization of sensor data |
| 4.8 | Build Patient Profile page | 4.2 + API | Patient management |
| 4.9 | Build Analysis page — trigger analysis, view results | 4.2 + API | Run ML inference from UI |
| 4.10 | Polish UI, responsive design, dark mode | 4.3–4.9 | Production-ready UI |

### Phase 5: Integration & Hardware Testing (Weeks 9–11)

> **⚠ Hardware required**: Full sensor array must be assembled and functional.

| # | Task | Depends On | Output |
|---|---|---|---|
| 5.1 | Assemble complete hardware prototype (solder, mount sensors in bra) | All hardware components | Working wearable prototype |
| 5.2 | Flash firmware with real sensor drivers (disable `USE_SIM_MODE`) | 5.1 | ESP32 reading real sensors |
| 5.3 | End-to-end test: ESP32 → receiver_logger → CSV → ML pipeline → backend → frontend | 5.2 + Phases 1–4 | Full pipeline validation |
| 5.4 | Collect real baseline data from test participants | 5.3 | Real healthy baseline dataset |
| 5.5 | Re-train models on real data | 5.4 | Updated models tuned to real sensor characteristics |
| 5.6 | Calibration routines (sensor drift compensation, ambient correction) | 5.4 | Calibration parameters |

### Phase 6: Polish, Documentation & Presentation (Weeks 11–12)

| # | Task | Depends On | Output |
|---|---|---|---|
| 6.1 | Write hardware setup documentation + wiring diagrams | Phase 5 | `docs/hardware_setup.md` |
| 6.2 | Write API documentation (auto-generated from FastAPI + manual) | Phase 3 | `docs/api_reference.md` |
| 6.3 | Write model documentation | Phase 2 | `docs/model_documentation.md` |
| 6.4 | Docker Compose for full stack deployment | All phases | One-command startup |
| 6.5 | Create PDF report generation script | Phase 3 | Professional scan reports |
| 6.6 | Final testing, bug fixes, performance optimization | All phases | Release-ready system |

---

## 7. ML/AI Pipeline Design

### 7.1 Pipeline Architecture

```
 Raw CSV ──► Validation ──► Cleaning ──► Windowing ──► Feature Extraction ──► Scaling ──► Model ──► Risk Score
                                              │                                              │
                                         (30s window,                                   Isolation Forest
                                          5s step)                                      or XGBoost
                                                                                             │
                                                                                    ┌────────▼────────┐
                                                                                    │  anomaly_score   │
                                                                                    │  (0.0 – 1.0)     │
                                                                                    │       │          │
                                                                                    │  risk_level:     │
                                                                                    │  LOW (< 0.3)     │
                                                                                    │  MODERATE (0.3-0.6)│
                                                                                    │  HIGH (0.6-0.8)  │
                                                                                    │  CRITICAL (> 0.8)│
                                                                                    └─────────────────┘
```

### 7.2 Model Strategy

**Phase 1 — Unsupervised (Isolation Forest)**:
- Train only on healthy/normal data
- Anomalies are samples that the forest isolates quickly (short path length)
- Score = normalized anomaly score (0–1)
- **Advantage**: No labeled anomaly data needed; works from day one

**Phase 2 — Semi-supervised (Autoencoder)** (optional):
- Neural network trained to reconstruct normal sensor patterns
- Anomalies produce high reconstruction error
- Better at capturing complex non-linear patterns

**Phase 3 — Supervised (XGBoost/Random Forest)** (when labeled data exists):
- Train on confirmed healthy vs. confirmed abnormal readings
- Output: probability of anomaly + classification
- **Advantage**: More accurate with good labels; can learn specific anomaly patterns

### 7.3 Risk Score Computation

```python
risk_score_raw = anomaly_model.score(features)     # 0.0 – 1.0
risk_score_calibrated = calibrate(risk_score_raw)   # Adjusted for prevalence / thresholds

# Composite scoring (weighted multi-modal)
w_temp = 0.40   # Temperature is the strongest single indicator
w_press = 0.30  # Pressure/stiffness is second
w_imp = 0.20    # Bioimpedance adds specificity
w_asym = 0.10   # Asymmetry bonus

composite = (w_temp * temp_anomaly_score +
             w_press * pressure_anomaly_score +
             w_imp * impedance_anomaly_score +
             w_asym * asymmetry_score)

risk_percentage = int(composite * 100)  # 0–100
```

---

## 7A. Simulation vs Production — Complete Data Architecture

> This section answers: Where does simulation fit? Do we train a separate model? How does data flow differ? What are notebooks vs scripts for?

### 7A.1 The Two Parallel Worlds

You are correct — there are **two distinct data paths** that feed into the **same processing pipeline**. The key design principle is:

> **The pipeline code is IDENTICAL. Only the data source changes.**

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        SIMULATION PATH (Now)                           ║
║                                                                        ║
║   [ simulate_data.py ]                                                 ║
║         │                                                              ║
║         ▼                                                              ║
║   data/raw/simulated/           ← CSVs land here                      ║
║         │                                                              ║
║         ▼                                                              ║
║   ┌─────────────────────────────────────────────┐                      ║
║   │         SHARED PIPELINE (same code)         │                      ║
║   │  preprocess → features → scale → train/pred │                      ║
║   └─────────────────────┬───────────────────────┘                      ║
║                         ▼                                              ║
║   models/simulation/anomaly_model.joblib   ← "Simulation Model"       ║
║                         │                                              ║
║                         ▼                                              ║
║   Backend loads this model → serves predictions via API                ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                     PRODUCTION PATH (Later, with hardware)             ║
║                                                                        ║
║   [ Sensors ] → [ ESP32 ] → [ receiver_logger.py ]                    ║
║                                    │                                   ║
║                                    ▼                                   ║
║   data/raw/hardware/               ← CSVs land here                   ║
║                                    │                                   ║
║                                    ▼                                   ║
║   ┌─────────────────────────────────────────────┐                      ║
║   │         SHARED PIPELINE (same code)         │                      ║
║   │  preprocess → features → scale → train/pred │                      ║
║   └─────────────────────┬───────────────────────┘                      ║
║                         ▼                                              ║
║   models/production/anomaly_model.joblib   ← "Production Model"       ║
║                         │                                              ║
║                         ▼                                              ║
║   Backend loads this model → serves predictions via API                ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 7A.2 Yes — Train a Separate Simulation Model

**Q: Should we train a model on simulated data?**
**A: Yes, absolutely.** Here's why and how:

| Aspect | Simulation Model | Production Model |
|---|---|---|
| **Trained on** | Synthetic data from `simulate_data.py` | Real sensor data from hardware |
| **Purpose** | Proof of concept, pipeline validation, demo, development | Actual anomaly detection on real patients |
| **When created** | NOW (Phase 1–2, no hardware needed) | LATER (Phase 5, after hardware data collection) |
| **Accuracy** | Lower — synthetic patterns are simplified | Higher — trained on real physiological signals |
| **Stored at** | `ml/models/simulation/` | `ml/models/production/` |
| **Used by backend** | Yes — until production model exists | Yes — replaces simulation model |

**The simulation model is NOT throwaway.** It serves critical roles:
1. **Validates the entire pipeline** works end-to-end before hardware exists
2. **Lets you build and test the backend + frontend** with real predictions
3. **Establishes baseline metrics** that the production model must beat
4. **Acts as fallback** if production model hasn't been trained yet

### 7A.3 Data Storage Architecture — Clear Separation

```
ml/
├── data/
│   ├── raw/
│   │   ├── simulated/                    # ══ SIMULATION DATA ══
│   │   │   ├── healthy/                  # Simulated normal readings
│   │   │   │   ├── sim_session_001.csv
│   │   │   │   ├── sim_session_002.csv
│   │   │   │   └── ...
│   │   │   ├── anomaly/                  # Simulated anomalous readings
│   │   │   │   ├── sim_anomaly_001.csv   # (hotspot injected)
│   │   │   │   ├── sim_anomaly_002.csv   # (asymmetry injected)
│   │   │   │   └── ...
│   │   │   └── manifest.json             # Metadata: labels, params used
│   │   │
│   │   └── hardware/                     # ══ REAL SENSOR DATA ══
│   │       ├── healthy/                  # Confirmed healthy subjects
│   │       │   ├── hw_session_001.csv
│   │       │   └── ...
│   │       ├── clinical/                 # Clinical data (if available)
│   │       │   └── ...
│   │       └── manifest.json             # Metadata: subject ID, conditions
│   │
│   ├── processed/
│   │   ├── simulated/                    # Feature matrices from sim data
│   │   │   ├── features.parquet
│   │   │   └── splits/
│   │   │       ├── train.parquet
│   │   │       ├── val.parquet
│   │   │       └── test.parquet
│   │   └── hardware/                     # Feature matrices from real data
│   │       ├── features.parquet
│   │       └── splits/
│   │           ├── train.parquet
│   │           ├── val.parquet
│   │           └── test.parquet
│   │
│   └── registry.json                     # Tracks all datasets + versions
│
├── models/
│   ├── simulation/                       # ══ SIMULATION MODEL ══
│   │   ├── anomaly_model.joblib
│   │   ├── scaler.joblib
│   │   ├── feature_columns.json          # Exact feature names/order
│   │   ├── training_metadata.json        # When trained, on what data, metrics
│   │   └── evaluation/
│   │       ├── metrics.json
│   │       └── plots/                    # ROC, confusion matrix, etc.
│   │
│   ├── production/                       # ══ PRODUCTION MODEL ══
│   │   ├── anomaly_model.joblib
│   │   ├── scaler.joblib
│   │   ├── feature_columns.json
│   │   ├── training_metadata.json
│   │   └── evaluation/
│   │       ├── metrics.json
│   │       └── plots/
│   │
│   └── active/                           # ══ ACTIVE MODEL (symlink) ══
│       └── -> ../simulation/             # Points to whichever model is deployed
│                                         # Switch to ../production/ when ready
```

### 7A.4 The Three Pipeline Modes — Training, Batch Prediction, Real-Time

The pipeline code lives in `ml/pipelines/` as **Python scripts**. The notebooks **call the same code** but add exploration and visualization on top.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODE 1: TRAINING (Notebook + Script)             │
│                                                                     │
│  Input:  data/raw/{simulated OR hardware}/*.csv                    │
│  Steps:  preprocess.py → feature_engineering.py → train_anomaly.py │
│  Output: models/{simulation OR production}/anomaly_model.joblib    │
│                                                                     │
│  Run via notebook: notebooks/01_train_simulation_model.ipynb        │
│                    notebooks/02_train_production_model.ipynb         │
│  Run via CLI:      python run_pipeline.py train --source simulated │
│                    python run_pipeline.py train --source hardware   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    MODE 2: BATCH PREDICTION (Script)                │
│                                                                     │
│  Input:  A CSV file (from any source)                              │
│  Steps:  preprocess.py → feature_engineering.py → predictor.py     │
│  Output: Risk scores for each window in the CSV                    │
│                                                                     │
│  Run via CLI:      python run_pipeline.py predict --input file.csv │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    MODE 3: REAL-TIME INFERENCE (Backend Service)    │
│                                                                     │
│  Input:  Live sensor JSON from ESP32 (via API or WebSocket)        │
│  Steps:  Same preprocess + feature code, but on streaming windows  │
│  Output: Risk score returned via API response                      │
│                                                                     │
│  Run via:  FastAPI backend (backend/app/services/ml_service.py)    │
│            Loads model from models/active/ at startup               │
└─────────────────────────────────────────────────────────────────────┘
```

### 7A.5 Notebooks vs Scripts — Who Does What

| Concern | Notebooks (`ml/notebooks/`) | Scripts (`ml/pipelines/`) |
|---|---|---|
| **Purpose** | Exploration, visualization, training with commentary | Automated, reproducible execution |
| **When used** | During development, experimentation, one-off training | Production pipeline, CI/CD, backend inference |
| **Who calls whom** | Notebooks **import from** `ml/pipelines/` and `ml/utils/` | Scripts are standalone, called via CLI |
| **Training** | ✅ Yes — interactive training with plots and analysis | ✅ Yes — automated training via `run_pipeline.py` |
| **Inference** | ❌ No — notebooks don't do real-time inference | ✅ Yes — backend uses scripts for prediction |

**Critical rule**: Notebooks NEVER contain unique pipeline logic. All logic lives in `ml/pipelines/` and `ml/utils/`. Notebooks just call those functions and add visualization.

```python
# Example: notebooks/01_train_simulation_model.ipynb (Cell 1)
# This notebook IMPORTS from the shared pipeline code

from ml.pipelines.preprocess import load_and_clean
from ml.pipelines.feature_engineering import extract_features
from ml.pipelines.train_anomaly import train_isolation_forest
from ml.utils.visualization import plot_anomaly_scores

# Cell 2: Load simulated data
df = load_and_clean(source="simulated")  # reads from data/raw/simulated/

# Cell 3: Extract features (same function used everywhere)
features_df = extract_features(df)

# Cell 4: Train  
model, scaler, metrics = train_isolation_forest(features_df)
# Saves to models/simulation/

# Cell 5: Visualize (notebook-only value-add)
plot_anomaly_scores(features_df, model)
```

### 7A.6 Notebook Structure

```
ml/notebooks/
├── 01_train_simulation_model.ipynb    # Train model on simulated data
│   ├── Load data from data/raw/simulated/
│   ├── Call preprocess.py functions
│   ├── Call feature_engineering.py functions
│   ├── Call train_anomaly.py functions
│   ├── Evaluate with evaluate.py functions
│   ├── Visualize results (plots, heatmaps)
│   └── Save model to models/simulation/
│
├── 02_train_production_model.ipynb    # Train model on real sensor data
│   ├── Load data from data/raw/hardware/
│   ├── Same pipeline functions as notebook 01
│   ├── Compare metrics vs simulation model
│   └── Save model to models/production/
│
├── 03_eda_simulated.ipynb             # Explore simulated data distributions
├── 04_eda_hardware.ipynb              # Explore real sensor data (when available)
├── 05_feature_exploration.ipynb       # Analyze which features matter most
└── 06_model_comparison.ipynb          # Compare sim vs prod model performance
```

### 7A.7 Production Inference Pipeline (Real-Time)

When the system is deployed, the production flow is:

```
Sensors → Firmware → API → Storage → Features → Model → Risk Score
   │         │        │       │          │         │         │
   │     ESP32     FastAPI  Database  Same code  Active   Response
   │    main.cpp   POST     readings  as train   model    to frontend
   │               /readings  table    pipeline  .joblib
```

The backend's `ml_service.py` replicates the exact same steps from the training notebooks, but as a Python function:

```python
# backend/app/services/ml_service.py (simplified)
class MLService:
    def __init__(self):
        # Load the ACTIVE model (simulation or production)
        model_dir = settings.ML_MODEL_PATH / "active"
        self.model = joblib.load(model_dir / "anomaly_model.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.feature_cols = json.load(open(model_dir / "feature_columns.json"))

    def predict(self, raw_readings: list[dict]) -> RiskScore:
        # Step 1: Same preprocess as training
        df = preprocess(raw_readings)
        # Step 2: Same feature extraction as training
        features = extract_features(df)
        # Step 3: Same scaling as training
        X = self.scaler.transform(features[self.feature_cols])
        # Step 4: Model prediction
        score = -self.model.score_samples(X)
        return compute_risk_score(score)
```

### 7A.8 Model Lifecycle — When Each Model Is Used

```
Timeline:
─────────────────────────────────────────────────────────►

Phase 1-2          Phase 3-4           Phase 5             Phase 6
(No hardware)      (Building app)      (Hardware ready)     (Production)
     │                  │                    │                   │
     ▼                  ▼                    ▼                   ▼
Generate           Backend loads        Collect real         Production
simulated    ──►   SIMULATION     ──►   sensor data    ──►  model replaces
data               model for            Train PROD          simulation
Train SIM          predictions          model               model
model              & dashboard                              
                                        Compare metrics     active/ → production/
active/ → simulation/                   If prod > sim:
                                        active/ → production/
```

| Phase | Active Model | Data Source | What Happens |
|---|---|---|---|
| 1–2 | `simulation` | `data/raw/simulated/` | Generate fake data, train sim model |
| 3–4 | `simulation` | Same | Backend + frontend use sim model for dev/demo |
| 5 | `simulation` → `production` | `data/raw/hardware/` | Collect real data, train prod model, compare, switch |
| 6+ | `production` | Real sensors | Production deployment |

### 7A.9 Database Tables — Separation of Concerns

In the backend database, readings are tagged by source but stored in the **same table**:

```sql
-- Single table, source-tagged (NOT separate tables)
CREATE TABLE sensor_readings (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(20) NOT NULL,        -- 'simulated' | 'hardware'
    -- Sensor values (same schema regardless of source)
    t1_c FLOAT, t2_c FLOAT, t3_c FLOAT, t4_c FLOAT,
    p1_raw INT, p2_raw INT, p3_raw INT, p4_raw INT,
    p5_raw INT, p6_raw INT, p7_raw INT, p8_raw INT,
    -- Future: impedance channels
    z1_ohm FLOAT, z2_ohm FLOAT, z3_ohm FLOAT, z4_ohm FLOAT,
    -- Metadata
    ambient_temp FLOAT,
    ambient_humidity FLOAT
);

-- Same risk_scores table for both
CREATE TABLE risk_scores (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    model_version VARCHAR(50) NOT NULL,  -- 'simulation_v1' | 'production_v1'
    overall_score FLOAT,
    risk_level VARCHAR(20),
    computed_at TIMESTAMPTZ
);
```

**Why NOT separate tables?** Because:
- The schema is identical — same sensors, same columns
- Queries and API endpoints work the same regardless of source
- You can filter by `source` when needed
- The `model_version` on risk scores tracks which model made each prediction

### 7A.10 Summary — Answers to Your Questions

| Question | Answer |
|---|---|
| Should we train a model on simulated data? | **Yes.** It's the simulation model — your working system before hardware. |
| Should we have a separate simulation dataset? | **Yes.** `data/raw/simulated/` is separate from `data/raw/hardware/`. |
| Separate tables for sim vs real data? | **No.** Same table, with a `source` column to tag origin. |
| Does simulated data need a training notebook? | **Yes.** `01_train_simulation_model.ipynb` — trains on sim data. |
| Does real data need a separate training notebook? | **Yes.** `02_train_production_model.ipynb` — trains on real data. |
| Are the pipelines the same? | **Yes.** Both notebooks import from the same `ml/pipelines/` code. |
| Where does simulation data come in? | At the **data generation** step — replaces sensors+firmware. |
| How does real-time prediction work? | Backend script imports same pipeline functions, runs on live data. |
| Which model does the backend use? | Whatever `models/active/` points to (simulation first, production later). |

---

## 8. API Design

### 8.1 REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register new user |
| `POST` | `/api/auth/login` | Login → JWT token |
| `GET` | `/api/patients` | List patients |
| `POST` | `/api/patients` | Create patient |
| `GET` | `/api/patients/{id}` | Get patient details |
| `PUT` | `/api/patients/{id}` | Update patient |
| `POST` | `/api/sessions` | Start a new scan session |
| `GET` | `/api/sessions` | List sessions (filterable) |
| `GET` | `/api/sessions/{id}` | Get session details |
| `PATCH` | `/api/sessions/{id}/complete` | Mark session complete |
| `POST` | `/api/readings/bulk` | Ingest raw sensor readings (CSV/JSON) |
| `GET` | `/api/readings?session_id=X` | Get readings for a session |
| `POST` | `/api/analyze` | Run ML analysis on a session |
| `GET` | `/api/risk-scores?session_id=X` | Get risk scores for a session |
| `GET` | `/api/risk-scores/latest?patient_id=X` | Get latest risk score for patient |
| `GET` | `/api/device/health` | Check connected device status |
| `WS` | `/ws/live` | Real-time sensor data stream |

### 8.2 Key Response Models

```json
// Risk Score Response
{
  "session_id": "uuid",
  "patient_id": "uuid",
  "timestamp": "2026-03-16T10:00:00Z",
  "overall_risk_score": 42,
  "risk_level": "MODERATE",
  "breakdown": {
    "temperature_score": 55,
    "pressure_score": 35,
    "impedance_score": 28,
    "asymmetry_score": 50
  },
  "top_features": [
    {"name": "temp_LminusR_mean", "importance": 0.23, "value": 1.2},
    {"name": "t2_c_slope", "importance": 0.18, "value": 0.05}
  ],
  "recommendation": "Moderate risk detected. Consider follow-up with clinical breast exam."
}
```

---

## 9. Frontend Dashboard Design

### 9.1 Pages & Key Components

| Page | Purpose | Key Components |
|---|---|---|
| **Dashboard** | At-a-glance overview | Risk gauge (0–100), latest readings cards, trend mini-charts, alerts banner |
| **Live Monitor** | Real-time sensor feed | WebSocket-driven multi-line charts (temp, pressure, impedance), connection status |
| **Session History** | Browse past scans | Sortable/filterable table, risk level badges, date range picker |
| **Session Detail** | Deep dive into one session | Full time-series charts, breast heatmap, risk score breakdown, SHAP feature chart |
| **Patient Profile** | Patient management | Demographics form, scan history timeline, risk trend over sessions |
| **Analysis** | Run new analysis | Upload CSV or select session, trigger analysis, view results |
| **Settings** | Configuration | Device settings, alert thresholds, sensor calibration offsets |

### 9.2 Breast Heatmap Visualization

The heatmap component overlays sensor readings on a schematic breast diagram:
- 4 temperature zones per breast (color: blue → yellow → red)
- 4 pressure zones per breast (size/opacity indicates stiffness)
- Impedance zones (contour overlay)
- Left-Right asymmetry highlighting

---

## 10. Deployment & DevOps

### 10.1 Docker Compose Setup

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: bra_logger
      POSTGRES_USER: bra_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: ./backend
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    environment:
      DATABASE_URL: postgresql+asyncpg://bra_user:${DB_PASSWORD}@db:5432/bra_logger
      ML_MODEL_PATH: /app/ml_models/
    volumes:
      - ./backend:/app
      - ./ml/models:/app/ml_models:ro
    ports:
      - "8000:8000"
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  pgdata:
```

---

## 11. Testing Strategy

| Layer | Tool | Focus |
|---|---|---|
| **Firmware** | PlatformIO + Unity | Sensor read functions, JSON serialization, mux channel selection |
| **ML Pipeline** | pytest | Feature extraction correctness, model loading, prediction shape, edge cases (NaN handling) |
| **Backend** | pytest + httpx (async) | API endpoint behavior, database operations, ML service integration, auth flow |
| **Frontend** | Vitest + React Testing Library | Component rendering, API mocking, user interactions |
| **E2E** | Playwright or Cypress | Full user flows: login → create patient → start session → view results |
| **Hardware** | Manual + scripts | Sensor connectivity, data integrity, Wi-Fi stability |

---

## 12. Ethical, Regulatory & Legal Considerations

| Concern | Approach |
|---|---|
| **Not a medical device (yet)** | All UI and reports must state: "This system is a screening aid and does not provide medical diagnosis. Consult a healthcare professional for any concerns." |
| **Data privacy** | Encrypt patient data at rest and in transit. Follow data minimization principles. No data sharing without consent. |
| **Informed consent** | Any human data collection requires informed consent forms and (for research) IRB approval. |
| **Bias** | Acknowledge that the system's accuracy may vary across demographics (breast size, skin tone, age). Document limitations. |
| **Clinical validation** | The system's risk scores must be validated against clinical outcomes before any deployment in medical settings. |

---

## Summary — First Steps Checklist

1. **Order hardware** (ESP32, DS18B20 ×4, FSR 402 ×8, CD74HC4067, 10kΩ + 4.7kΩ resistors, breadboard, wires, battery + TP4056)
2. **Refactor firmware** into modular `firmware/` folder structure
3. **Build advanced simulation** (`data_collection/simulate_data.py`) with anomaly injection
4. **Generate 500+ simulated sessions** (healthy + anomalous)
5. **Build ML pipeline** — preprocess → features → Isolation Forest → evaluation
6. **Initialize FastAPI backend** with database models and ML service
7. **Initialize React frontend** with dashboard layout
8. **Iterate** — collect real data → retrain → improve

---

> **This document is a living plan.** Update it as the project evolves, decisions are made, and hardware arrives.
