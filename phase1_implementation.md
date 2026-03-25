# Phase 1 Implementation — BRA Logger

This document covers what was implemented in **Phase 1 (Foundation & Data Collection)** based on `implementation.md`, what each file does, how the simulator works, what packages to install, and what to do next.

> **Scope reminder**: Phase 1 builds the *data foundation* (data contract + session storage + simulation + receiver logger). It does **not** train models yet.

---

## Critical analysis of the Phase 1 problem statement

Phase 1 is solving a very specific engineering problem:

- **You need a single, stable “data contract”** (columns, units, ordering, bounds) that is identical whether data comes from:
  - the simulaton,
  - the device/ESP32 receiver logger
- **You need repeatable, diverse time-series sessions** to validate downstream ML windowing/feature pipelines and to support development of later backend/frontend phases even before hardware exists.
- **You need reliable session storage and metadata** so later pipeline code can:
  - trace exactly how a session was generated/recorded,
  - filter by label/source/session type,
  - reproduce simulations (seed + parameters), and
  - avoid “mystery CSVs” with unknown provenance.

The biggest risk in projects like this is *schema drift* (simulator vs hardware CSVs differ) and *data quality issues* (missing fields, spikes, corrupted rows). This phase therefore focuses on strict schema + robust logging + realistic simulation noise/artifacts.

---

## Folder structure (Phase 1 relevant)

Phase 1 work focused on improving **data ingestion + simulation + shared schema**.

- `data_collection/`
  - **Data collection** (hardware receiver) + **simulation** + **session management**
- `ml/`
  - Early “shared pipeline” utilities (preprocessing + windowing + features). These are not fully used until Phase 2, but they need the Phase 1 schema to remain stable.

---

## Packages to install (Phase 1)

`requirements.txt` in the repo root was updated to contain **Phase 1 only** packages (names only, no versions):

- `numpy`
- `pandas`
- `pyyaml`
- `requests`
- `tqdm`

Install them with:

```bash
pip install -r requirements.txt
```

---

## Data contract (shared schema)

### `data_collection/schema.py`

This is the **single source of truth** for the Phase 1 raw-row schema:

- **Column groups** (temperature, pressure, impedance, environment, imu)
- **Canonical CSV column order**: `CSV_COLUMNS`
- **Bounds** used for clipping obviously invalid values (`BOUNDS`)
- **Sanitization helpers** used by both simulator and receiver:
  - `ensure_columns(...)`: fills missing keys with `None`
  - `normalize_row_types_inplace(...)`: coerces types (pressure → int; other sensors → float)
  - `clip_row_inplace(...)`: clips values into plausible ranges
  - `now_iso_ms()`: standardized PC-side ISO timestamp

Why this matters:
- Downstream processing (Phase 2+) becomes simpler because files are predictable.
- Hardware firmware can evolve; your receiver logger can still enforce the contract.

---

## Session storage and metadata

### `data_collection/session_manager.py`

This module manages **session lifecycle** and makes raw data storage organized and auditable.

What it does:

- Creates the directory structure: `.../<base_dir>/<label>/...`
- Writes one CSV per session
- Maintains a global manifest: `<base_dir>/manifest.json`
- **New in Phase 1 update**:
  - Writes a per-session sidecar metadata file next to the CSV:
    - `.../some_session.csv`
    - `.../some_session.meta.json`

Why both a manifest and per-session metadata?

- `manifest.json` helps you quickly list/search sessions in bulk.
- `*.meta.json` keeps a session self-describing if you copy it elsewhere.

Each session record includes:
- `session_id`, `label`, `source`
- `start_time`, `end_time`
- `num_rows`
- `metadata` (free-form: simulator seed/profile, device URL, health check, etc.)

---

## Advanced simulator (Phase 1 deliverable)

### `data_collection/simulate_data.py`

This script generates **physiologically-inspired multi-sensor time-series sessions** matching the same schema as hardware ingestion.

#### What it produces

For each session it writes:

- A session CSV with columns in exactly `CSV_COLUMNS`
- A `*.meta.json` for that session (seed, profile, anomaly type, hz, duration)
- Updates a `manifest.json` in the base directory

Output goes to (by default):
- `ml/data/raw/simulated/healthy/*.csv`
- `ml/data/raw/simulated/anomaly/*.csv`
- `ml/data/raw/simulated/manifest.json`

#### How it simulates data (high level)

The simulator generates a session as a combination of:

- **A per-session subject profile** (inter-session variability)
  - baseline offsets per channel (temp/pressure/impedance)
  - noise levels per modality
  - drift rates per modality
  - breathing rate and phase
- **Baseline signals** (healthy physiology + wearable artifacts)
  - temperature: baseline + slow drift + breathing micro-oscillation + random-walk drift + noise
  - pressure: baseline + breathing compression + contact variation + drift + noise
  - impedance: baseline + slow oscillation + drift + noise
  - environment: slow ambient temperature/humidity variation
  - IMU: near-rest gravity vector with micro-motion + occasional burst movement
- **Artifacts**
  - **motion bursts**: short IMU spikes (and pressure perturbations) to mimic posture adjustment
  - **missingness/dropouts (optional)**: random channels set to `null` at low probability to harden preprocessing
- **Anomaly injection** (anomaly sessions only)
  - An anomaly starts at a random onset fraction and lasts for a random duration fraction.
  - Injection uses a **smooth cosine ramp** mask (gradual onset and offset, not step functions).

#### Anomaly types supported

Based on your plan:
- `temp_hotspot`: +0.5 to +2.0°C on 1–2 temp channels
- `pressure_lump`: +30% to +65% on 1–2 pressure channels
- `impedance_drop`: −35% to −60% on 1–2 impedance channels
- `asymmetry`: persistent L-R difference (temp and/or pressure)
- `combined`: co-occurrence (temp hotspot + pressure lump + impedance drop)

#### Key configuration file

The simulator reads `data_collection/config.yaml` for:
- sampling rate and duration
- anomaly type probabilities and magnitudes
- timing ranges
- motion burst probability
- dropout probability

---

## Real-device receiver logger (Phase 1 deliverable)

### `data_collection/receiver_logger.py`

This script polls the ESP32 HTTP endpoint and writes hardware sessions to CSV using the same schema.

Key Phase 1 improvements:

- **Schema enforcement**: every received JSON reading is:
  - completed using `ensure_columns(...)`
  - type-normalized using `normalize_row_types_inplace(...)`
  - clipped using `clip_row_inplace(...)`
- **Stable header order**: the session CSV header is always `CSV_COLUMNS` (even if a device row is missing some keys)
- **Session metadata** now includes:
  - URL, health URL, `device_health_ok`
  - `hz`, `duration_s`

Default output (from config):
- `ml/data/raw/hardware/healthy/*.csv` (or `clinical` based on CLI label choice)
- `ml/data/raw/hardware/manifest.json`

---

## Shared pipeline code (Phase 1 foundation for Phase 2)

Even though model training is Phase 2, Phase 1 must ensure the pipeline code has a stable schema.

### `ml/pipelines/preprocess.py`

This module already existed; Phase 1 updated it to **import the canonical column groups** from `data_collection/schema.py` instead of duplicating them.

What it does:
- loads one or many session CSVs
- ensures sensor columns exist (fills missing)
- casts dtypes
- clips to plausible bounds
- imputes missing values

### `ml/utils/windowing.py`

Sliding window helpers:
- 30s window @ 2 Hz → 60 rows
- 5s step @ 2 Hz → 10 rows

This will be used directly in Phase 2 feature extraction and modeling.

### `ml/utils/features.py`

Feature building blocks (stats + FFT-based spectral features + asymmetry/cross-modal features).

Note: Phase 1 doesn’t require you to install `scipy` anymore (requirements were narrowed to Phase 1). If you start running Phase 2 feature extraction that imports `scipy`, you’ll add it back in Phase 2 requirements.

---

## What every Phase 1 file is for (summary)

- **`requirements.txt`**: Phase 1-only dependencies (names only).
- **`data_collection/schema.py` (new)**: canonical schema + bounds + sanitization helpers.
- **`data_collection/session_manager.py` (updated)**: creates sessions, writes CSV + per-session metadata + manifest.
- **`data_collection/simulate_data.py` (updated)**: generates realistic simulated sessions + anomaly injection.
- **`data_collection/receiver_logger.py` (updated)**: polls ESP32 endpoint and writes hardware sessions with schema enforcement.
- **`data_collection/config.yaml` (updated)**: simulator + receiver configuration (sampling, output dirs, anomaly settings, missingness/motion).
- **`ml/pipelines/preprocess.py` (updated)**: preprocessing uses canonical schema definitions.

---

## How to run Phase 1 (commands you run)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Generate simulated datasets

Uses defaults from `data_collection/config.yaml` (e.g., 500 healthy + 150 anomaly):

```bash
python -m data_collection.simulate_data
```

What happens:
- Reads `data_collection/config.yaml` for:
  - `sampling.hz`, `sampling.session_duration_s`
  - `simulation.sessions.healthy`, `simulation.sessions.anomaly` (unless overridden by CLI)
  - anomaly probabilities/magnitudes/timing, motion + dropout settings
- Generates **two sets** of sessions:
  - **Healthy** sessions written to `ml/data/raw/simulated/healthy/`
  - **Anomaly** sessions written to `ml/data/raw/simulated/anomaly/`
- For each session it writes:
  - `*.csv` (one row per sample, columns in **exact `CSV_COLUMNS` order**)
  - `*.meta.json` (per-session metadata: seed, hz, duration, anomaly type, subject profile)
- Updates/creates `ml/data/raw/simulated/manifest.json` (dataset-level index of sessions).

What you should check:
- **Files created**:
  - `ml/data/raw/simulated/manifest.json` exists and grows as sessions are generated.
  - `ml/data/raw/simulated/healthy/` contains many `simulated_healthy_*.csv` files (and matching `.meta.json`).
  - `ml/data/raw/simulated/anomaly/` contains many `simulated_anomaly_*.csv` files (and matching `.meta.json`).
- **Row counts**:
  - Each CSV should have approximately \(hz \times duration\) rows (default: \(2.0 \times 300 = 600\) rows).
  - Each session file should have **one header row** + the data rows.
- **Schema sanity** (open any one generated CSV):
  - The header includes `pc_time_iso`, `ts_ms`, `source`, `label`, then sensor columns (`t*`, `p*`, `z*`, ambient, accel).
  - `source` should be `simulated`.
  - `label` should be `healthy` or `anomaly_<type>`.
- **Common issues**:
  - If you get `ModuleNotFoundError: tqdm/yaml/numpy/pandas`: run `pip install -r requirements.txt`.
  - If no files appear, confirm `output.simulated_dir` in `data_collection/config.yaml` points to `ml/data/raw/simulated`.

Override counts:

```bash
python -m data_collection.simulate_data --healthy 50 --anomaly 20
```

Dry-run (prints a few rows, writes nothing):

```bash
python -m data_collection.simulate_data --dry-run
```

What happens (dry-run):
- Generates two short in-memory sessions (healthy + a `temp_hotspot` anomaly) and prints the **first 5 rows** of each.
- **Does not write any files** and does not modify manifests.

What you should check (dry-run):
- The printed rows include all keys from the schema (timestamps, `source`, `label`, and sensor channels).
- Values look plausible:
  - temps in the mid-30°C range
  - pressure within 0–4095
  - impedance in the hundreds to low thousands of Ω
  - accel `z` near ~1.0 at rest
- If the anomaly sample is `temp_hotspot`, one temperature channel should be noticeably higher than baseline during the anomaly region (not necessarily within the first 5 printed rows, depending on onset timing).

### 3) Record hardware sessions (when device is available)

```bash
python -m data_collection.receiver_logger --label healthy --sessions 1
```

What happens:
- Loads config from `data_collection/config.yaml`.
- Polls the ESP32 `device.url` (or `--url`) at `sampling.hz` for `sampling.session_duration_s`.
- Creates a new session under `ml/data/raw/hardware/healthy/` and writes:
  - `*.csv` with header in **exact `CSV_COLUMNS` order**
  - `*.meta.json` sidecar with metadata:
    - url, health_url, device_health_ok, hz, duration_s, session_num
- Updates/creates `ml/data/raw/hardware/manifest.json`.
- If the device JSON is missing fields, it fills them with `null`, normalizes types, and clips out-of-range values before writing the row.

What you should check:
- **Connectivity**:
  - If you see “device health check failed”, it means the `/health` endpoint didn’t return HTTP 200 (often Wi‑Fi / wrong IP / firmware not serving that route).
  - If you see repeated fetch errors, the `/data` endpoint isn’t reachable or isn’t returning valid JSON.
- **Files created**:
  - A new `hardware_healthy_*.csv` appears in `ml/data/raw/hardware/healthy/` (and matching `.meta.json`).
  - `ml/data/raw/hardware/manifest.json` is created/updated.
- **Row counts**:
  - If the session runs full duration, expect about \(hz \times duration\) rows (default ~600).
  - If it aborts early (many consecutive errors), the CSV will be shorter (still valid).
- **Schema**:
  - `source` should be `hardware`.
  - `label` should be `healthy`.
  - Header should match the simulator’s header order (important for Phase 2+).

Override URL:

```bash
python -m data_collection.receiver_logger --url http://192.168.4.1/data --sessions 1
```

---

## What to do next (Phase 1 → Phase 2 bridge)

After generating sessions, your immediate next steps are:

- **Verify raw data layout**
  - Confirm CSVs and `manifest.json` are created under:
    - `ml/data/raw/simulated/healthy/`
    - `ml/data/raw/simulated/anomaly/`
- **Start Phase 2 preprocessing + windowing**
  - Use `ml/pipelines/preprocess.py` to load and clean
  - Use `ml/utils/windowing.py` to create 30s windows with 5s step
- **Phase 2 dependency update**
  - When you begin Phase 2 feature extraction (`ml/utils/features.py` uses SciPy), add Phase 2 packages back (e.g., `scipy`, `scikit-learn`, etc.) in a Phase 2 requirements update.

---

## Notes / design decisions (important)

- **We intentionally enforce schema at ingestion time** (receiver logger + simulator) so later pipeline steps can be simpler and more reliable.
- **We write both manifest and per-session metadata** because real-world datasets inevitably get moved around.
- **We simulated realistic imperfections** (drift, motion artifacts, dropouts) so Phase 2 preprocessing and models don’t overfit to “too-clean” synthetic data.
---

## High Level Overview

This section breaks down Phase 1 in plain language with analogies so you can understand the "why" and "what" without getting lost in code details.

---

### What is Phase 1, really?

Imagine you want to build a car factory. Before you can assemble a single car, you need to:

1. **Design the parts list** — agree that every car will have exactly 4 wheels, 1 engine, 2 mirrors, etc. (this is the **schema**).
2. **Build the assembly line** — conveyors, stamping machines, painting booths (this is the **session manager** + **simulator** + **receiver logger**).
3. **Set up quality control** — check that every car coming off the line meets spec (this is **preprocessing** + **validation**).

You haven't built a single car yet (that's Phase 2 where ML models are trained). But when Phase 2 starts, everything is ready — the factory floor is set up, the machines work, and you can pump out cars (data) on demand.

**Phase 1 = "Set up the data factory so that Phase 2 (AI/ML) has reliable, consistent data to learn from."**

---

### The big picture: how data flows in Phase 1

```
┌──────────────────────────────────────────────────────────────┐
│                      PHASE 1 DATA FLOW                       │
│                                                              │
│   config.yaml ─── controls everything ───────────┐          │
│                                                   ▼          │
│   ┌─────────────────┐     ┌─────────────────┐              │
│   │  simulate_data   │     │ receiver_logger  │              │
│   │  (fake sensor    │     │ (real sensor     │              │
│   │   data factory)  │     │  data recorder)  │              │
│   └────────┬────────┘     └────────┬────────┘              │
│            │                        │                        │
│            │  Both use schema.py    │                        │
│            │  to format every row   │                        │
│            │  identically           │                        │
│            ▼                        ▼                        │
│   ┌─────────────────────────────────────────┐               │
│   │          session_manager.py              │               │
│   │  (organises CSVs + metadata + manifest)  │               │
│   └────────────────────┬────────────────────┘               │
│                         │                                    │
│                         ▼                                    │
│   ml/data/raw/simulated/healthy/*.csv                        │
│   ml/data/raw/simulated/anomaly/*.csv                        │
│   ml/data/raw/hardware/healthy/*.csv  (future)               │
│                         │                                    │
│                         ▼                                    │
│   ┌─────────────────────────────────────────┐               │
│   │       preprocess.py  (quality control)   │               │
│   │       windowing.py   (slice into chunks) │               │
│   │       features.py    (measure properties)│  ← Phase 2   │
│   └─────────────────────────────────────────┘    uses these  │
└──────────────────────────────────────────────────────────────┘
```

---

### What each file does (with analogies)

#### 1. `data_collection/schema.py` — The Blueprint

**Analogy**: Think of a standardised shipping container. No matter who packs it (the simulator or the real device), every container has the same dimensions, the same labelled slots, and the same rules about what can go where.

What it defines:
- **21 sensor columns** in a fixed order: 4 temperature, 8 pressure, 4 impedance, 2 environment, 3 accelerometer.
- **4 metadata columns**: timestamp, millisecond counter, data source, and label.
- **Valid ranges** (bounds) for every sensor — e.g., skin temperature must be between 20°C and 45°C. Anything outside gets clipped.
- **Helper functions** used by both the simulator and receiver:
  - `ensure_columns()` — "Did you forget a slot? Let me add it with a blank value."
  - `normalize_row_types_inplace()` — "Pressure must be an integer, temperature must be a decimal."
  - `clip_row_inplace()` — "That reading is impossibly high/low, capping it."
  - `now_iso_ms()` — "What time is it right now, in a standard format?"

**Why it matters**: Without this, the simulator might produce columns in one order and the device in another. Then your ML pipeline breaks. The schema prevents that.

---

#### 2. `data_collection/config.yaml` — The Control Panel

**Analogy**: A thermostat + settings panel on your wall. You turn the dials here — "How fast should we sample? How many sessions? What kind of anomalies should occur?" — and every other file reads from this panel instead of having its own hard-coded numbers.

Key settings:
- **Sampling**: 2 Hz (2 readings per second), 300 seconds per session (5 minutes).
- **Device**: the ESP32's URL for when real hardware is connected.
- **Output paths**: where to store simulated vs hardware CSV files.
- **Channel definitions**: which columns belong to temperature, pressure, impedance, etc., and which are left-breast vs right-breast.
- **Simulation parameters**: how many healthy/anomaly sessions, anomaly type probabilities (35% temp hotspot, 25% pressure lump, etc.), magnitude ranges, timing, motion artifact probability, and dropout probability.

**Why it matters**: Changing one number here (e.g., session duration from 300s to 600s) automatically propagates to both the simulator and the receiver logger. No hunting through code.

---

#### 3. `data_collection/session_manager.py` — The Filing Clerk

**Analogy**: Imagine a meticulous office clerk. Every time a recording session happens (real or simulated), the clerk:
1. Opens a new folder in the right cabinet (`healthy/` or `anomaly/`).
2. Creates a fresh CSV file and writes the header row.
3. Accepts one row at a time and writes it to the CSV.
4. When the session ends, stamps a receipt (`*.meta.json`) next to the CSV with details like: when did it start, when did it end, how many rows, what kind of session.
5. Updates the master log book (`manifest.json`) so you can quickly look up all sessions ever recorded.

Key pieces:
- **`SessionManager`** class: you tell it "I want to record a healthy simulated session" and it handles all the folder creation, naming, and bookkeeping.
- **`_SessionWriter`** (internal): the actual CSV writer — it writes rows, flushes periodically (every 50 rows) to prevent data loss if the program crashes.
- **Context manager** (`with mgr.new_session(...) as sess:`): ensures the CSV is always properly closed and metadata is always written, even if something goes wrong.

**Why it matters**: Without this, you'd end up with hundreds of CSV files scattered around with no record of what they contain, how they were generated, or whether they're healthy or anomalous. The filing clerk keeps everything organised and traceable.

---

#### 4. `data_collection/simulate_data.py` — The Flight Simulator

**Analogy**: Before pilots fly real planes, they train in a flight simulator that recreates realistic conditions — turbulence, weather, instrument failures. Similarly, this script creates fake sensor data that behaves like real breast tissue sensors would, including realistic noise and imperfections.

How it generates a session:
1. **Creates a "subject profile"** — random but consistent baseline offsets per sensor, noise levels, breathing rate. This means each simulated session is slightly different, just like readings would vary between real people.
2. **Generates healthy baseline signals**:
   - **Temperature**: ~34°C skin surface + slow drift + breathing micro-oscillation + random noise.
   - **Pressure**: ~1500 ADC reading + breathing compression cycles + contact variation + noise.
   - **Impedance**: ~800 Ω + slow oscillation + drift + noise.
   - **Environment**: room temperature (~22°C) and humidity (~50%) with slow variation.
   - **IMU**: gravity on the z-axis (~1.0g) + tiny micro-motion + occasional motion bursts (posture shifts).
3. **For anomaly sessions, injects one of five anomaly types**:
   - **Temperature hotspot**: 1–2 temperature sensors rise 0.5–2°C (simulating increased blood flow near a mass).
   - **Pressure lump**: 1–2 pressure sensors increase 30–65% (simulating a hard lump under the sensor).
   - **Impedance drop**: 1–2 impedance sensors fall 35–60% (simulating more vascular tissue).
   - **Asymmetry**: left breast reads noticeably different from right breast in temperature or pressure.
   - **Combined**: multiple anomaly types co-occurring (the most severe).
4. **Uses a smooth ramp** (cosine curve) so anomalies fade in and out gradually, not as sudden jumps — this trains the AI to detect realistic onset patterns.
5. **Adds realistic artifacts**: occasional motion bursts in the IMU, random sensor dropouts (null values), pressure perturbations from posture shifts.

**Why it matters**: You don't have hardware yet. This simulator is your entire data supply chain. Without it, Phase 2 (ML training) has nothing to learn from.

---

#### 5. `data_collection/receiver_logger.py` — The Live Recorder

**Analogy**: A court stenographer — sits there, listens to everything being said (sensor readings from the ESP32), and transcribes it into the official record (CSV files) in real time.

How it works:
1. Polls the ESP32's HTTP endpoint (e.g., `http://192.168.4.1/data`) at 2 Hz.
2. Each JSON response is run through the schema's sanitisation: fill missing columns, fix types, clip out-of-range values.
3. Writes the cleaned row to a CSV via the session manager.
4. If the device stops responding (20 consecutive errors), it gracefully stops and saves whatever was collected.
5. Handles Ctrl+C cleanly (flushes the CSV, writes metadata).

**Why it matters**: When you eventually build the real hardware, this script is ready to go. It speaks the exact same "language" (schema) as the simulator, so your ML pipeline won't know or care whether data came from a real bra or a simulation. That's the whole point of the shared data contract.

---

#### 6. `ml/pipelines/preprocess.py` — The Quality Inspector

**Analogy**: A quality control station at the end of an assembly line. Every CSV that arrives gets inspected:
- Missing columns? → Added with blank/NaN values.
- Wrong data type (text where a number should be)? → Converted or marked as missing.
- Temperature reading of 500°C? → Clipped to 45°C.
- Gaps in the data (null values)? → Filled in using linear interpolation between neighbouring values.

Key functions:
- `load_csv()` — loads and cleans a single session file.
- `load_sessions()` — loads all CSVs from a directory.
- `load_and_clean()` — loads everything for a given source ("simulated" or "hardware") into one big DataFrame ready for ML.

**Why it matters**: Raw data is messy — even simulated data can have intentional dropouts (to test robustness). This module ensures the ML pipeline always receives clean, consistent, gap-free data.

---

#### 7. `ml/utils/windowing.py` — The Cookie Cutter

**Analogy**: You have a long strip of dough (a 5-minute session = 600 rows). You stamp out cookies that are each 30 seconds long (60 rows). You slide the cutter forward by 5 seconds (10 rows) between stamps, so cookies overlap. Each cookie is one "sample" for the ML model.

Key functions:
- `create_windows()` — takes one session DataFrame → returns a list of 60-row window DataFrames.
- `windows_from_sessions()` — does the same across many sessions at once.

Default: 30-second window, 5-second step, 2 Hz → each window is 60 rows, and a 5-minute session produces approximately 55 windows.

**Why it matters**: ML models can't process a 5-minute stream directly. They need fixed-size input. The windowing converts variable sessions into uniform chunks.

---

#### 8. `ml/utils/features.py` — The Lab Technician

**Analogy**: A lab technician receives each cookie-cutter sample and runs a battery of tests on it: mean temperature, how much it fluctuates, is there a trend, what frequencies appear, is the left side different from the right?

Feature types computed per window:
- **Time-domain** (13 features per channel): mean, std, min, max, range, median, IQR, skewness, kurtosis, slope, energy, first-derivative mean and std.
- **Spectral** (4 features per channel): dominant frequency, spectral entropy, spectral energy, spectral centroid.
- **Cross-channel** (spatial): left-vs-right difference (mean, std, max, fraction exceeding thresholds), hotspot deviation, cross-modal correlation.

**Note**: This file imports `scipy`, which is not in Phase 1 requirements. It exists as scaffolding for Phase 2 and will run once `scipy` is installed.

---

#### 9. `ml/utils/visualization.py` — The Dashboard Painter

**Analogy**: A graphic designer who turns raw numbers into charts you can actually look at.

Provides:
- `plot_session()` — 4-panel chart showing temperature, pressure, impedance, and environment over time for one session.
- `plot_anomaly_scores()` — histogram of anomaly scores with optional healthy/anomaly colour coding and threshold line.

**Note**: Imports `matplotlib`, not in Phase 1 requirements. Ready for use once Phase 2 packages are installed.

---

#### 10. `requirements.txt` — The Shopping List

Lists exactly 5 packages needed for Phase 1: `numpy`, `pandas`, `pyyaml`, `requests`, `tqdm`. Nothing more, nothing less. Install with `pip install -r requirements.txt`.

---

### Is Phase 1 complete?

| Aspect | Status | Details |
|---|---|---|
| **Code** | ✅ Complete | All 10 files are fully written. Schema, session manager, simulator, receiver logger, config, preprocessing, windowing, features, visualization, and requirements are all present and internally consistent. |
| **Data** | ⚠️ Not generated yet | The `ml/data/raw/simulated/healthy/` and `ml/data/raw/simulated/anomaly/` directories are empty (only `.gitkeep`). You need to run `python -m data_collection.simulate_data` to produce the CSV datasets. |
| **Phase 2 scaffolding** | ℹ️ Exists but needs extra packages | `features.py` needs `scipy` and `visualization.py` needs `matplotlib`. These are intentionally omitted from Phase 1 requirements since they're Phase 2 dependencies. The code is there and correct — it just won't import until you install those packages. |

**Bottom line**: Phase 1 code is 100% written. To fully "complete" Phase 1, run the simulator: `python -m data_collection.simulate_data` to generate your dataset, then you're ready for Phase 2.
