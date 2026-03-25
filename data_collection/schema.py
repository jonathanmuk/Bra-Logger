from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


TEMP_COLS = ("t1_c", "t2_c", "t3_c", "t4_c")
PRESSURE_COLS = ("p1_raw", "p2_raw", "p3_raw", "p4_raw", "p5_raw", "p6_raw", "p7_raw", "p8_raw")
IMPEDANCE_COLS = ("z1_ohm", "z2_ohm", "z3_ohm", "z4_ohm")
ENV_COLS = ("ambient_temp_c", "ambient_humidity_pct")
IMU_COLS = ("accel_x", "accel_y", "accel_z")

META_COLS = ("pc_time_iso", "ts_ms", "source", "label")
SENSOR_COLS = TEMP_COLS + PRESSURE_COLS + IMPEDANCE_COLS + ENV_COLS + IMU_COLS

# Canonical CSV column order for all sessions (simulated and hardware).
CSV_COLUMNS = META_COLS + SENSOR_COLS


@dataclass(frozen=True)
class ColumnBounds:
    lo: float
    hi: float


BOUNDS: dict[str, ColumnBounds] = {
    # Temperature (°C): plausible skin surface range
    **{c: ColumnBounds(20.0, 45.0) for c in TEMP_COLS},
    # Pressure (12-bit ADC)
    **{c: ColumnBounds(0.0, 4095.0) for c in PRESSURE_COLS},
    # Impedance (Ω): wide, clips only obvious spikes
    **{c: ColumnBounds(10.0, 5000.0) for c in IMPEDANCE_COLS},
    # Ambient
    "ambient_temp_c": ColumnBounds(5.0, 45.0),
    "ambient_humidity_pct": ColumnBounds(0.0, 100.0),
    # IMU accel (g): wide for motion artifacts, still bounded
    "accel_x": ColumnBounds(-8.0, 8.0),
    "accel_y": ColumnBounds(-8.0, 8.0),
    "accel_z": ColumnBounds(-8.0, 8.0),
}


def now_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ensure_columns(row: dict[str, Any], *, fill_value: float | None = None) -> dict[str, Any]:
    """Ensure the row has all canonical keys. Missing keys are filled with `fill_value`."""
    out = dict(row)
    for k in CSV_COLUMNS:
        if k not in out:
            out[k] = fill_value
    return out


def clip_row_inplace(row: dict[str, Any]) -> None:
    """Clip numeric values to plausible bounds (best-effort)."""
    for k, bounds in BOUNDS.items():
        if k not in row:
            continue
        v = row[k]
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv < bounds.lo:
            row[k] = bounds.lo
        elif fv > bounds.hi:
            row[k] = bounds.hi


def normalize_row_types_inplace(row: dict[str, Any]) -> None:
    """
    Make row values JSON/CSV friendly and consistent:
    - sensor floats remain floats
    - pressure channels become ints if possible
    """
    for k in TEMP_COLS + IMPEDANCE_COLS + ENV_COLS + IMU_COLS:
        if k in row and row[k] is not None:
            try:
                row[k] = float(row[k])
            except (TypeError, ValueError):
                row[k] = None
    for k in PRESSURE_COLS:
        if k in row and row[k] is not None:
            try:
                row[k] = int(float(row[k]))
            except (TypeError, ValueError):
                row[k] = None


def validate_row_keys(row: dict[str, Any], *, required: Iterable[str] = CSV_COLUMNS) -> list[str]:
    """Return a list of missing keys (does not raise)."""
    missing: list[str] = []
    for k in required:
        if k not in row:
            missing.append(k)
    return missing

