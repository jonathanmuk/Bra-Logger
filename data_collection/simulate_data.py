"""
data_collection/simulate_data.py
=================================
Advanced synthetic data generator for the BRA Logger project.

How it works
------------
1. Reads simulation parameters from ``data_collection/config.yaml``.
2. For each requested session it produces a time-series CSV that mimics
   what the ESP32 firmware would stream.
3. "Healthy" sessions use physiologically-plausible baseline signals
   with natural drift, breathing-rate micro-variation, and sensor noise.
4. "Anomaly" sessions inject one or more of five anomaly types at a
   random onset time and duration.  The injection is smooth (ramp up /
   ramp down) so the model learns gradual onset, not step functions.

Anomaly types
-------------
temp_hotspot  – one or two temperature sensors rise 0.5–2 °C above
                baseline, simulating increased angiogenesis near a mass.
pressure_lump – one or two pressure channels increase 30–65 %, simulating
                a hard mass under the sensor.
impedance_drop– one or two impedance channels drop 35–60 %, simulating
                higher tissue vascularity.
asymmetry     – a persistent cross-breast temperature or pressure
                imbalance exceeding clinical thresholds.
combined      – two of the above co-occur (most severe).

Usage
-----
  # Activate venv first, then:
  python -m data_collection.simulate_data           # uses config.yaml defaults
  python -m data_collection.simulate_data --healthy 200 --anomaly 50
  python -m data_collection.simulate_data --dry-run  # prints 5 sample rows only
"""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from data_collection.session_manager import SessionManager
from data_collection.schema import CSV_COLUMNS, clip_row_inplace, ensure_columns, normalize_row_types_inplace

# ── Project root (so relative paths resolve correctly) ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "data_collection" / "config.yaml"


# =============================================================================
# Config loader
# =============================================================================

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Signal generators — building blocks
# =============================================================================

class SignalGenerator:
    """
    Generates one continuous multi-channel physiological signal for a session.

    The generator is seeded per-session so results are reproducible given
    the same session index.
    """

    COLUMNS = list(CSV_COLUMNS)

    def __init__(self, cfg: dict, hz: float = 2.0, duration_s: float = 300.0, seed: int = 0):
        self.cfg = cfg
        self.hz = hz
        self.duration_s = duration_s
        self.n_samples = int(hz * duration_s)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.t = np.linspace(0, duration_s, self.n_samples)
        self.profile = self._make_subject_profile()

    # ── Public ──────────────────────────────────────────────────────────────

    def healthy_session(self) -> list[dict[str, Any]]:
        """Generate one healthy baseline session."""
        temps = self._base_temperatures()
        pressures = self._base_pressures()
        impedances = self._base_impedances()
        env = self._environment()
        imu = self._imu_with_micro_motion()
        return self._assemble_rows("healthy", temps, pressures, impedances, env, imu)

    def anomaly_session(self, anomaly_type: str | None = None) -> list[dict[str, Any]]:
        """
        Generate one anomaly session.

        Parameters
        ----------
        anomaly_type: override which anomaly is injected (default: random choice
                      weighted by config probabilities).
        """
        if anomaly_type is None:
            anomaly_type = self._choose_anomaly_type()

        temps = self._base_temperatures()
        pressures = self._base_pressures()
        impedances = self._base_impedances()
        env = self._environment()
        imu = self._imu_with_micro_motion()

        # Inject anomaly
        onset, offset = self._anomaly_window()
        mask = self._smooth_mask(onset, offset)

        if anomaly_type == "temp_hotspot":
            temps = self._inject_temp_hotspot(temps, mask)
        elif anomaly_type == "pressure_lump":
            pressures = self._inject_pressure_lump(pressures, mask)
        elif anomaly_type == "impedance_drop":
            impedances = self._inject_impedance_drop(impedances, mask)
        elif anomaly_type == "asymmetry":
            # Pick whether asymmetry manifests in temp or pressure (or both) based on config.
            which = self.rng.choices(["temp", "pressure", "both"], weights=[0.55, 0.35, 0.10], k=1)[0]
            if which in ("temp", "both"):
                temps = self._inject_temp_asymmetry(temps, mask)
            if which in ("pressure", "both"):
                pressures = self._inject_pressure_asymmetry(pressures, mask)
        elif anomaly_type == "combined":
            temps = self._inject_temp_hotspot(temps, mask)
            pressures = self._inject_pressure_lump(pressures, mask)
            impedances = self._inject_impedance_drop(impedances, mask)

        rows = self._assemble_rows(f"anomaly_{anomaly_type}", temps, pressures, impedances, env, imu)
        return rows

    # ── Baseline signals ────────────────────────────────────────────────────

    def _base_temperatures(self) -> np.ndarray:
        """
        Shape: (n_samples, 4)
        Models skin surface temperature with:
          - slow circadian drift (sinusoidal, ~20s period in simulation)
          - small inter-sensor offset (sensors placed at different quadrants)
          - breathing-rate micro-oscillation (~0.2 Hz)
          - Gaussian noise
        """
        base = float(self.cfg["channels"]["temperature"]["normal_baseline"])
        n, t = self.n_samples, self.t
        # slow body-temp drift (simulation time-compressed)
        drift = 0.15 * np.sin(2 * math.pi * t / 25.0 + self.profile["circadian_phase"])
        # breathing oscillation
        breath = 0.05 * np.sin(2 * math.pi * self.profile["breath_hz"] * t)
        # sensor drift (random walk)
        drift_rw = self._random_walk(n, sigma=self.profile["temp_drift_sigma"])
        channels = []
        for i in range(4):
            # each sensor has a fixed small offset (anatomy)
            offset = self.profile["temp_offsets"][i]
            # per-sensor micro-variation
            micro = 0.03 * np.sin(2 * math.pi * t / (7.0 + i))
            noise = self.np_rng.normal(0, self.profile["temp_noise_sigma"], n)
            channels.append(base + offset + drift + breath + micro + drift_rw[:, i] + noise)
        return np.column_stack(channels)  # (N, 4)

    def _base_pressures(self) -> np.ndarray:
        """
        Shape: (n_samples, 8)
        Models bra compression force with breathing-driven oscillation
        and sensor-specific contact variation.
        """
        base = float(self.cfg["channels"]["pressure"]["normal_baseline"])
        n, t = self.n_samples, self.t
        # breathing compression oscillation
        breath = self.profile["pressure_breath_amp"] * np.sin(2 * math.pi * (self.profile["breath_hz"] + 0.03) * t)
        drift_rw = self._random_walk(n, sigma=self.profile["pressure_drift_sigma"], channels=8)
        channels = []
        for i in range(8):
            offset = self.profile["pressure_offsets"][i]
            micro = 20.0 * np.sin(2 * math.pi * t / (5.0 + i * 0.5))
            noise = self.np_rng.normal(0, self.profile["pressure_noise_sigma"], n)
            raw = base + offset + breath + micro + drift_rw[:, i] + noise
            raw = np.clip(raw, 0, 4095)
            channels.append(raw)
        pressures = np.column_stack(channels)  # (N, 8)
        # Couple occasional motion artifacts into pressure (looser contact, posture change).
        pressures = self._inject_motion_artifacts_into_pressure(pressures)
        return pressures

    def _base_impedances(self) -> np.ndarray:
        """
        Shape: (n_samples, 4)
        Tissue impedance at ~50 kHz — relatively stable with small drift.
        """
        base = float(self.cfg["channels"]["impedance"]["normal_baseline"])
        n, t = self.n_samples, self.t
        drift_rw = self._random_walk(n, sigma=self.profile["impedance_drift_sigma"])
        channels = []
        for i in range(4):
            offset = self.profile["impedance_offsets"][i]
            drift = 10.0 * np.sin(2 * math.pi * t / 60.0 + i)
            noise = self.np_rng.normal(0, self.profile["impedance_noise_sigma"], n)
            channels.append(base + offset + drift + drift_rw[:, i] + noise)
        return np.column_stack(channels)  # (N, 4)

    def _environment(self) -> dict[str, np.ndarray]:
        """Ambient temperature and humidity — slowly varying."""
        n, t = self.n_samples, self.t
        amb_temp = (22.0 + self.rng.uniform(-2, 2)
                    + 0.2 * np.sin(2 * math.pi * t / 120)
                    + self.np_rng.normal(0, 0.1, n))
        humidity = (50.0 + self.rng.uniform(-5, 5)
                    + 1.0 * np.sin(2 * math.pi * t / 90)
                    + self.np_rng.normal(0, 0.5, n))
        return {"temp": amb_temp, "humidity": humidity}

    def _imu_with_micro_motion(self) -> dict[str, np.ndarray]:
        """
        Accelerometer baseline — gravity (≈1 g on z) + small micro-motion,
        plus occasional short bursts to mimic posture adjustment.
        """
        n, t = self.n_samples, self.t
        x = self.np_rng.normal(0.0, 0.01, n)
        y = self.np_rng.normal(0.0, 0.01, n)
        z = self.np_rng.normal(1.0, 0.01, n)

        # Rare short bursts (motion artifacts).
        p_burst = float(self.cfg.get("simulation", {}).get("motion_burst_probability", 0.25))
        if self.rng.random() < p_burst:
            burst_count = self.rng.randint(1, 3)
            for _ in range(burst_count):
                center = self.rng.randint(int(0.05 * n), int(0.95 * n))
                width = self.rng.randint(int(0.02 * n), int(0.06 * n))
                amp = self.rng.uniform(0.2, 0.8)
                burst = amp * np.exp(-0.5 * ((np.arange(n) - center) / max(1, width)) ** 2)
                x += burst * self.np_rng.normal(1.0, 0.1, n)
                y += burst * self.np_rng.normal(1.0, 0.1, n)
                z += burst * self.np_rng.normal(0.2, 0.05, n)

        # Very slow orientation drift (device micro-shift).
        x += 0.02 * np.sin(2 * math.pi * t / 180.0)
        y += 0.02 * np.cos(2 * math.pi * t / 210.0)
        return {"x": x, "y": y, "z": z}

    # ── Anomaly injection ────────────────────────────────────────────────────

    def _choose_anomaly_type(self) -> str:
        types = list(self.cfg["simulation"]["anomaly_types"].keys())
        weights = [self.cfg["simulation"]["anomaly_types"][k] for k in types]
        return self.rng.choices(types, weights=weights, k=1)[0]

    def _anomaly_window(self) -> tuple[int, int]:
        """Returns (onset_sample_idx, offset_sample_idx)."""
        n = self.n_samples
        onset_frac  = self.rng.uniform(*self.cfg["simulation"]["onset_fraction"])
        dur_frac    = self.rng.uniform(*self.cfg["simulation"]["duration_fraction"])
        onset  = int(n * onset_frac)
        offset = min(n, onset + int((n - onset) * dur_frac))
        return onset, offset

    def _smooth_mask(self, onset: int, offset: int, ramp_samples: int = 10) -> np.ndarray:
        """
        Creates a (N,) mask in [0,1] with smooth ramp-up and ramp-down.
        This teaches the model to expect gradual onset, not step changes.
        """
        mask = np.zeros(self.n_samples, dtype=float)
        mask[onset:offset] = 1.0
        ramp = min(ramp_samples, max(1, offset - onset))
        # Cosine ramp-up / ramp-down to avoid sharp edges.
        ramp_up = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ramp))
        mask[onset : onset + ramp] = ramp_up
        ramp_down = ramp_up[::-1]
        mask[offset - ramp : offset] = ramp_down
        return mask

    def _inject_temp_hotspot(self, temps: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Raise 1–2 temperature sensors by delta_c during the anomaly window."""
        temps = temps.copy()
        delta_c = self.rng.uniform(*self.cfg["simulation"]["anomaly_magnitude"]["temp_delta_c"])
        affected = self.rng.sample(range(4), k=self.rng.randint(1, 2))
        for ch in affected:
            temps[:, ch] += delta_c * mask
        return temps

    def _inject_pressure_lump(self, pressures: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Raise 1–2 pressure sensors by a fraction (hard mass simulation)."""
        pressures = pressures.copy()
        frac = self.rng.uniform(*self.cfg["simulation"]["anomaly_magnitude"]["pressure_delta"])
        affected = self.rng.sample(range(8), k=self.rng.randint(1, 2))
        for ch in affected:
            pressures[:, ch] += pressures[:, ch].mean() * frac * mask
            pressures[:, ch] = np.clip(pressures[:, ch], 0, 4095)
        return pressures

    def _inject_impedance_drop(self, impedances: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Drop 1–2 impedance channels (increased vascularity simulation)."""
        impedances = impedances.copy()
        frac = self.rng.uniform(*self.cfg["simulation"]["anomaly_magnitude"]["impedance_drop"])
        affected = self.rng.sample(range(4), k=self.rng.randint(1, 2))
        for ch in affected:
            drop = impedances[:, ch].mean() * frac
            impedances[:, ch] -= drop * mask
            impedances[:, ch] = np.maximum(impedances[:, ch], 50)  # floor
        return impedances

    def _inject_temp_asymmetry(self, temps: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create a persistent left-right temperature difference (thermography criterion)."""
        temps = temps.copy()
        delta = self.rng.uniform(*self.cfg["simulation"]["anomaly_magnitude"]["asymmetry_temp"])
        # Left breast = channels 0,1; right = 2,3
        for ch in (0, 1):
            temps[:, ch] += delta * mask
        return temps

    def _inject_pressure_asymmetry(self, pressures: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create a persistent left-right pressure difference (stiffness asymmetry)."""
        pressures = pressures.copy()
        frac = self.rng.uniform(*self.cfg["simulation"]["anomaly_magnitude"]["asymmetry_press"])
        # Left breast = channels 0..3; right = 4..7
        left_mean = pressures[:, :4].mean()
        delta = left_mean * frac
        for ch in range(4):
            pressures[:, ch] += delta * mask
        pressures = np.clip(pressures, 0, 4095)
        return pressures

    # ── Realism helpers ─────────────────────────────────────────────────────

    def _make_subject_profile(self) -> dict[str, Any]:
        """
        A lightweight "subject profile" to introduce inter-session variability.
        This is where baseline offsets, noise levels, and drift characteristics vary.
        """
        # Temperature offsets (°C) per sensor.
        temp_offsets = [0.05 * i + self.rng.uniform(-0.10, 0.10) for i in range(4)]
        # Pressure offsets (ADC) per sensor; higher variance due to contact.
        pressure_offsets = [30.0 * i + self.rng.uniform(-80, 80) for i in range(8)]
        # Impedance offsets (Ω) per sensor.
        impedance_offsets = [self.rng.uniform(-80, 80) for _ in range(4)]

        return {
            "circadian_phase": self.rng.uniform(0.0, 2 * math.pi),
            "breath_hz": self.rng.uniform(0.16, 0.30),  # ~10–18 breaths/min
            "temp_offsets": temp_offsets,
            "pressure_offsets": pressure_offsets,
            "impedance_offsets": impedance_offsets,
            # Noise (measurement) sigma
            "temp_noise_sigma": self.rng.uniform(0.015, 0.035),
            "pressure_noise_sigma": self.rng.uniform(6.0, 14.0),
            "impedance_noise_sigma": self.rng.uniform(2.0, 6.0),
            # Drift (random walk) sigma per step
            "temp_drift_sigma": self.rng.uniform(0.0005, 0.0025),
            "pressure_drift_sigma": self.rng.uniform(0.4, 2.0),
            "impedance_drift_sigma": self.rng.uniform(0.05, 0.25),
            "pressure_breath_amp": self.rng.uniform(50.0, 120.0),
        }

    def _random_walk(self, n: int, sigma: float, channels: int = 4) -> np.ndarray:
        steps = self.np_rng.normal(0.0, sigma, (n, channels))
        return np.cumsum(steps, axis=0)

    def _inject_motion_artifacts_into_pressure(self, pressures: np.ndarray) -> np.ndarray:
        """
        Model brief posture shifts: pressure channels simultaneously dip or spike.
        Triggered by the same probability as IMU bursts, but implemented independently
        to avoid tight coupling.
        """
        p = float(self.cfg.get("simulation", {}).get("motion_burst_probability", 0.25))
        if self.rng.random() >= p:
            return pressures

        n = pressures.shape[0]
        artifact = np.zeros((n,), dtype=float)
        burst_count = self.rng.randint(1, 2)
        for _ in range(burst_count):
            center = self.rng.randint(int(0.10 * n), int(0.90 * n))
            width = self.rng.randint(int(0.02 * n), int(0.05 * n))
            amp = self.rng.uniform(-220.0, 320.0)
            artifact += amp * np.exp(-0.5 * ((np.arange(n) - center) / max(1, width)) ** 2)

        pressures = pressures + artifact[:, None]
        return np.clip(pressures, 0, 4095)

    # ── Row assembly ─────────────────────────────────────────────────────────

    def _assemble_rows(
        self,
        label: str,
        temps: np.ndarray,
        pressures: np.ndarray,
        impedances: np.ndarray,
        env: dict,
        imu: dict,
    ) -> list[dict[str, Any]]:
        rows = []
        start_ts = time.time()
        for i in range(self.n_samples):
            ts_ms = int(i * (1000.0 / self.hz))
            row = {
                "pc_time_iso": datetime.fromtimestamp(
                    start_ts + i / self.hz, tz=timezone.utc
                ).isoformat(timespec="milliseconds"),
                "ts_ms": ts_ms,
                "source": "simulated",
                "label": label,
                # Temperature
                "t1_c": round(float(temps[i, 0]), 4),
                "t2_c": round(float(temps[i, 1]), 4),
                "t3_c": round(float(temps[i, 2]), 4),
                "t4_c": round(float(temps[i, 3]), 4),
                # Pressure
                "p1_raw": int(pressures[i, 0]),
                "p2_raw": int(pressures[i, 1]),
                "p3_raw": int(pressures[i, 2]),
                "p4_raw": int(pressures[i, 3]),
                "p5_raw": int(pressures[i, 4]),
                "p6_raw": int(pressures[i, 5]),
                "p7_raw": int(pressures[i, 6]),
                "p8_raw": int(pressures[i, 7]),
                # Impedance
                "z1_ohm": round(float(impedances[i, 0]), 2),
                "z2_ohm": round(float(impedances[i, 1]), 2),
                "z3_ohm": round(float(impedances[i, 2]), 2),
                "z4_ohm": round(float(impedances[i, 3]), 2),
                # Environment
                "ambient_temp_c":       round(float(env["temp"][i]), 2),
                "ambient_humidity_pct": round(float(env["humidity"][i]), 1),
                # IMU
                "accel_x": round(float(imu["x"][i]), 4),
                "accel_y": round(float(imu["y"][i]), 4),
                "accel_z": round(float(imu["z"][i]), 4),
            }
            # Optional missingness / sensor dropouts (helps harden preprocessing).
            miss_cfg = self.cfg.get("simulation", {}).get("missingness", {}) or {}
            p_drop = float(miss_cfg.get("dropout_probability", 0.0))
            if p_drop > 0 and self.rng.random() < p_drop:
                # Drop 1–3 random sensor channels (not metadata).
                k = self.rng.randint(1, 3)
                candidates = [c for c in self.COLUMNS if c not in ("pc_time_iso", "ts_ms", "source", "label")]
                for col in self.rng.sample(candidates, k=k):
                    row[col] = None

            row = ensure_columns(row, fill_value=None)
            normalize_row_types_inplace(row)
            clip_row_inplace(row)
            rows.append(row)
        return rows


# =============================================================================
# Runner
# =============================================================================

def run_simulation(
    cfg: dict,
    n_healthy: int,
    n_anomaly: int,
    output_base: str,
    dry_run: bool = False,
) -> None:
    """
    Generate all sessions and write CSVs.

    Parameters
    ----------
    cfg       : loaded config dict
    n_healthy : number of healthy sessions to generate
    n_anomaly : number of anomaly sessions to generate
    output_base : path like "ml/data/raw/simulated"
    dry_run   : if True, prints 5 rows and exits (no files written)
    """
    hz       = cfg["sampling"]["hz"]
    duration = cfg["sampling"]["session_duration_s"]

    if dry_run:
        gen = SignalGenerator(cfg, hz=hz, duration_s=30, seed=0)
        rows = gen.healthy_session()
        print("\n── DRY RUN: first 5 rows of a healthy session ──")
        for r in rows[:5]:
            print(r)
        print("\n── DRY RUN: first 5 rows of an anomaly (temp_hotspot) session ──")
        gen2 = SignalGenerator(cfg, hz=hz, duration_s=30, seed=1)
        rows2 = gen2.anomaly_session("temp_hotspot")
        for r in rows2[:5]:
            print(r)
        return

    simulated_dir = PROJECT_ROOT / output_base

    # Healthy sessions
    healthy_mgr = SessionManager(
        base_dir=simulated_dir, label="healthy", source="simulated"
    )
    print(f"\nGenerating {n_healthy} healthy sessions → {simulated_dir}/healthy/")
    for i in tqdm(range(n_healthy), unit="session"):
        gen = SignalGenerator(cfg, hz=hz, duration_s=duration, seed=i)
        rows = gen.healthy_session()
        with healthy_mgr.new_session(
            metadata={"seed": i, "hz": hz, "duration_s": duration, "profile": gen.profile},
            fieldnames=list(CSV_COLUMNS),
        ) as sess:
            for row in rows:
                sess.write_row(row)

    # Anomaly sessions
    anomaly_mgr = SessionManager(
        base_dir=simulated_dir, label="anomaly", source="simulated"
    )
    print(f"\nGenerating {n_anomaly} anomaly sessions → {simulated_dir}/anomaly/")
    for i in tqdm(range(n_anomaly), unit="session"):
        gen = SignalGenerator(cfg, hz=hz, duration_s=duration, seed=n_healthy + i)
        anomaly_rows = gen.anomaly_session()
        label = anomaly_rows[0]["label"]  # e.g. "anomaly_temp_hotspot"
        with anomaly_mgr.new_session(
            metadata={"seed": n_healthy + i, "hz": hz, "duration_s": duration, "type": label, "profile": gen.profile},
            fieldnames=list(CSV_COLUMNS),
        ) as sess:
            for row in anomaly_rows:
                sess.write_row(row)

    print(
        f"\n✓ Done. Generated {n_healthy} healthy + {n_anomaly} anomaly sessions "
        f"in {simulated_dir}"
    )


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    cfg = load_config()
    default_healthy = cfg["simulation"]["sessions"]["healthy"]
    default_anomaly = cfg["simulation"]["sessions"]["anomaly"]
    default_out     = cfg["output"]["simulated_dir"]

    p = argparse.ArgumentParser(
        description="Generate synthetic BRA Logger sensor data with optional anomaly injection."
    )
    p.add_argument("--healthy",  type=int, default=default_healthy,
                   help=f"Number of healthy sessions (default: {default_healthy})")
    p.add_argument("--anomaly",  type=int, default=default_anomaly,
                   help=f"Number of anomaly sessions (default: {default_anomaly})")
    p.add_argument("--output",   default=default_out,
                   help=f"Output base directory (default: {default_out})")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print sample rows without writing files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_config()
    run_simulation(
        cfg=config,
        n_healthy=args.healthy,
        n_anomaly=args.anomaly,
        output_base=args.output,
        dry_run=args.dry_run,
    )
