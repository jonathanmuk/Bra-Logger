"""
data_collection/receiver_logger.py
====================================
Polls the ESP32 HTTP /data endpoint and writes readings to CSV.
Runs in "device" mode (the real counterpart to simulate_data.py).

Usage
-----
  python -m data_collection.receiver_logger            # uses config.yaml
  python -m data_collection.receiver_logger --label healthy --sessions 1
  python -m data_collection.receiver_logger --url http://192.168.4.1/data
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import requests
import yaml

from data_collection.session_manager import SessionManager
from data_collection.schema import CSV_COLUMNS, clip_row_inplace, ensure_columns, normalize_row_types_inplace, now_iso_ms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "data_collection" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class DeviceLogger:
    """
    Polls the ESP32 /data endpoint at the configured rate and
    writes readings to CSV via SessionManager.

    The produced CSV has the same column schema as simulate_data.py,
    except `source` is "hardware" and `label` is whatever you pass in.
    """

    def __init__(self, cfg: dict, url: str | None = None, label: str = "healthy") -> None:
        self.cfg = cfg
        self.url = url or cfg["device"]["url"]
        self.timeout = cfg["device"]["timeout_s"]
        self.hz = cfg["sampling"]["hz"]
        self.interval = 1.0 / self.hz
        self.duration_s = cfg["sampling"]["session_duration_s"]
        self.label = label
        self.output_dir = PROJECT_ROOT / cfg["output"]["hardware_dir"]

        self._mgr = SessionManager(
            base_dir=self.output_dir,
            label=label,
            source="hardware",
        )
        self._stop = False

    def _health_check(self) -> bool:
        health_url = self.cfg["device"]["health_url"]
        try:
            r = requests.get(health_url, timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _fetch_reading(self) -> dict | None:
        """Fetch one JSON reading from the ESP32."""
        try:
            r = requests.get(self.url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # Tag with source and label (device may omit these).
            data["source"] = "hardware"
            data["label"] = self.label
            # If firmware doesn't provide an ISO timestamp, add PC time.
            data.setdefault("pc_time_iso", now_iso_ms())
            # If firmware doesn't provide monotonic timestamp, leave as None.
            data = ensure_columns(data, fill_value=None)
            normalize_row_types_inplace(data)
            clip_row_inplace(data)
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"[receiver_logger] Fetch error: {e}", file=sys.stderr)
            return None

    def record_session(self, session_num: int = 1) -> None:
        """Record one full session."""
        print(f"\n[receiver_logger] Starting session {session_num}")
        print(f"  Device URL : {self.url}")
        print(f"  Label      : {self.label}")
        print(f"  Duration   : {self.duration_s}s @ {self.hz} Hz")

        if not self._health_check():
            print("[receiver_logger] ⚠ Device health check failed — is the ESP32 connected?")

        n_samples = int(self.hz * self.duration_s)
        error_count = 0
        max_errors = 20

        health_ok = self._health_check()
        with self._mgr.new_session(
            metadata={
                "session_num": session_num,
                "url": self.url,
                "health_url": self.cfg["device"]["health_url"],
                "device_health_ok": bool(health_ok),
                "hz": self.hz,
                "duration_s": self.duration_s,
            },
            fieldnames=list(CSV_COLUMNS),
        ) as sess:
            for i in range(n_samples):
                if self._stop:
                    print("\n[receiver_logger] Interrupted — flushing session...")
                    break

                t0 = time.monotonic()
                row = self._fetch_reading()
                if row:
                    sess.write_row(row)
                    error_count = 0
                else:
                    error_count += 1
                    if error_count >= max_errors:
                        print(f"[receiver_logger] {max_errors} consecutive errors — aborting.", file=sys.stderr)
                        break

                # Maintain sample rate
                elapsed = time.monotonic() - t0
                sleep_t = self.interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        print(f"[receiver_logger] ✓ Session {session_num} complete — {sess.num_rows} rows")

    def handle_signal(self, signum, frame):
        self._stop = True


def _parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Real-device sensor logger for BRA Logger")
    p.add_argument("--url",      default=cfg["device"]["url"])
    p.add_argument("--label",    default="healthy", choices=["healthy", "clinical"])
    p.add_argument("--sessions", type=int, default=1, help="Number of sessions to record")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()
    cfg["device"]["url"] = args.url

    logger = DeviceLogger(cfg=cfg, url=args.url, label=args.label)
    signal.signal(signal.SIGINT, logger.handle_signal)

    for i in range(1, args.sessions + 1):
        logger.record_session(session_num=i)
        if logger._stop:
            break
