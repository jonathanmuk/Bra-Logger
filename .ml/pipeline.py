import glob
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn as sklearn
import pandas as pd
import numpy as np

HZ = 2
WINDOW_SEC = 30
STEP_SEC = 5

def make_windows(df, hz=HZ, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    win_len = int(window_sec * hz)
    step_len = int(step_sec * hz)

    windows = []
    for start in range(0, len(df) - win_len + 1, step_len):
        w = df.iloc[start:start + win_len]
        windows.append(w)
    return windows, win_len, step_len

def debug_windowing(csv_path):
    df = pd.read_csv(csv_path)

    # Basic checks
    required = ["ts_ms", "t1_c","t2_c","t3_c","t4_c","p1_raw","p2_raw","p3_raw","p4_raw","p5_raw","p6_raw","p7_raw","p8_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("ts_ms").reset_index(drop=True)

    windows, win_len, step_len = make_windows(df)

    total_samples = len(df)
    expected_windows = 0
    if total_samples >= win_len:
        expected_windows = ((total_samples - win_len) // step_len) + 1

    print("---- WINDOW DEBUG ----")
    print("HZ:", HZ)
    print("WINDOW_SEC:", WINDOW_SEC, "=> win_len:", win_len, "samples")
    print("STEP_SEC:", STEP_SEC, "=> step_len:", step_len, "samples")
    print("Total samples:", total_samples)
    print("Windows created:", len(windows))
    print("Expected windows:", expected_windows)

    # Verify window sizes
    bad = [i for i,w in enumerate(windows) if len(w) != win_len]
    print("Bad window sizes:", len(bad))

    # Verify step spacing using timestamps
    if len(windows) >= 2:
        ts0 = windows[0]["ts_ms"].iloc[0]
        ts1 = windows[1]["ts_ms"].iloc[0]
        print("First window start ts_ms:", ts0)
        print("Second window start ts_ms:", ts1)
        print("Delta start (ms):", ts1 - ts0, "Expected ~", int(STEP_SEC*1000))

    # Show sample of first window
    w0 = windows[0]
    print("\nFirst window preview (first 3 rows):")
    print(w0.head(3)[["ts_ms","t1_c","t2_c","t3_c","t4_c","p1_raw","p8_raw"]])

    print("\nLast window preview (last 3 rows):")
    print(windows[-1].tail(3)[["ts_ms","t1_c","t2_c","t3_c","t4_c","p1_raw","p8_raw"]])

    print("---- END DEBUG ----")
    debug_windowing() in main()

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib as joblib 


# ---------------------------
# CONFIG (edit if you want)
# ---------------------------
HZ = 2                    # sampling rate (2 Hz)
WINDOW_SEC = 30           # 30s windows
STEP_SEC = 5              # slide by 5s
CONTAMINATION = 0.03      # expected anomaly fraction (tune later)

TEMP_COLS = ["t1_c", "t2_c", "t3_c", "t4_c"]
P_COLS = ["p1_raw", "p2_raw", "p3_raw", "p4_raw", "p5_raw", "p6_raw", "p7_raw", "p8_raw"]
REQ_COLS = ["pc_time_iso", "ts_ms"] + TEMP_COLS + P_COLS

# Map sensors into left/right groups (adjust if your physical layout differs)
LEFT_T = ["t1_c", "t2_c"]
RIGHT_T = ["t3_c", "t4_c"]
LEFT_P = ["p1_raw", "p2_raw", "p3_raw", "p4_raw"]
RIGHT_P = ["p5_raw", "p6_raw", "p7_raw", "p8_raw"]


# ---------------------------
# HELPERS
# ---------------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Parse time
    df["pc_time_iso"] = pd.to_datetime(df["pc_time_iso"], errors="coerce")

    # Cast numeric columns
    for c in ["ts_ms"] + TEMP_COLS + P_COLS:
        df[c] = df[c].map(_safe_float)

    # Drop rows with no timestamp at all
    df = df.dropna(subset=["ts_ms"]).copy()

    # Sort by ts
    df = df.sort_values("ts_ms").reset_index(drop=True)

    # Basic cleanup: fill small gaps (optional)
    # Here we just forward-fill a little; you can change later.
    df[TEMP_COLS + P_COLS] = df[TEMP_COLS + P_COLS].ffill().bfill()

    return df


def feats_1d(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return {}

    dx = np.diff(x)
    t = np.arange(len(x))
    slope = np.polyfit(t, x, 1)[0]  # linear trend

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_slope": float(slope),
        f"{prefix}_energy": float(np.mean(x**2)),
        f"{prefix}_dmean": float(np.mean(dx)) if len(dx) else 0.0,
        f"{prefix}_dstd": float(np.std(dx)) if len(dx) else 0.0,
    }


def window_iter(df: pd.DataFrame, win_len: int, step_len: int):
    n = len(df)
    for start in range(0, n - win_len + 1, step_len):
        yield start, df.iloc[start:start + win_len]


def extract_window_features(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    win_len = int(WINDOW_SEC * HZ)
    step_len = int(STEP_SEC * HZ)

    rows = []
    for start_idx, w in window_iter(df, win_len, step_len):
        feat = {}

        # Per-channel features (12 channels)
        for c in TEMP_COLS + P_COLS:
            feat.update(feats_1d(w[c].values, c))

        # Spatial features (L-R difference)
        left_t = w[LEFT_T].mean(axis=1).values
        right_t = w[RIGHT_T].mean(axis=1).values
        feat.update(feats_1d(left_t - right_t, "temp_LminusR"))

        left_p = w[LEFT_P].mean(axis=1).values
        right_p = w[RIGHT_P].mean(axis=1).values
        feat.update(feats_1d(left_p - right_p, "press_LminusR"))

        # Hotspot features
        feat["temp_hotspot"] = float(w[TEMP_COLS].max(axis=1).mean() - w[TEMP_COLS].mean(axis=1).mean())
        feat["press_hotspot"] = float(w[P_COLS].max(axis=1).mean() - w[P_COLS].mean(axis=1).mean())

        # Metadata
        feat["source_file"] = source_file
        feat["start_ts_ms"] = int(w["ts_ms"].iloc[0])
        feat["start_pc_time"] = w["pc_time_iso"].iloc[0].isoformat() if pd.notna(w["pc_time_iso"].iloc[0]) else ""

        rows.append(feat)

    feats_df = pd.DataFrame(rows).dropna(axis=1, how="all")
    return feats_df


def train_and_score(feats_df: pd.DataFrame, out_dir: Path):
    meta_cols = ["source_file", "start_ts_ms", "start_pc_time"]
    feature_cols = [c for c in feats_df.columns if c not in meta_cols]

    X = feats_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=400,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xs)

    # Flip sign so: higher score = more abnormal
    anomaly_score = -model.score_samples(Xs)
    feats_df = feats_df.copy()
    feats_df["anomaly_score"] = anomaly_score

    out_dir.mkdir(parents=True, exist_ok=True)
    feats_df.to_csv(out_dir / "features_with_scores.csv", index=False)

    joblib.dump(
        {"scaler": scaler, "model": model, "feature_cols": feature_cols},
        out_dir / "anomaly_model.joblib"
    )

    top = feats_df.sort_values("anomaly_score", ascending=False).head(10)[
        ["source_file", "start_pc_time", "start_ts_ms", "anomaly_score"]
    ]

    print("\nSaved:")
    print(" -", out_dir / "features_with_scores.csv")
    print(" -", out_dir / "anomaly_model.joblib")
    print("\nTop 10 most abnormal windows:")
    print(top.to_string(index=False))


def main():
    # Find CSVs in project root (same as platformio.ini)
    csvs = sorted(glob.glob("bra_raw_*.csv"))
    if not csvs:
        raise SystemExit("No bra_raw_*.csv files found in this folder. Run receiver_logger.py first.")

    all_feats = []
    for fp in csvs:
        df = pd.read_csv(fp)
        df = validate_and_clean(df)
        feats = extract_window_features(df, source_file=fp)
        if len(feats) > 0:
            all_feats.append(feats)

    if not all_feats:
        raise SystemExit("No windows extracted. Need longer recordings (at least WINDOW_SEC seconds).")

    feats_df = pd.concat(all_feats, ignore_index=True)

    out_dir = Path("ml_out")
    train_and_score(feats_df, out_dir)

def main():
    csv_path = "bra_raw_sim_20260301_.csv"  # rmake sure this file exists
    debug_windowing(csv_path)

if __name__ == "__main__":

    main()

    # Quick debug of windowing logic (run this if you just want to verify windows are created correctly)
    # debug_windowing("bra_raw_sim_20240630_123456.csv")  # replace with your actual CSV path