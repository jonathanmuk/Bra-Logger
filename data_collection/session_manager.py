"""
data_collection/session_manager.py
====================================
Manages the lifecycle of a data collection session:
 - Creates the output directory structure
 - Writes / reads the session manifest (metadata JSON)
 - Provides a context manager for safe CSV writing

A "session" is one continuous recording: e.g., 5 minutes of a healthy
subject sitting still.  Each session produces one CSV file.

Usage
-----
  from data_collection.session_manager import SessionManager

  mgr = SessionManager(base_dir="ml/data/raw/simulated", label="healthy")
  with mgr.new_session(metadata={"subject": "sim_001"}) as session:
      session.write_row(row_dict)
  # CSV is flushed and manifest updated on exit
"""

from __future__ import annotations

import csv
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SessionInfo:
    """Immutable descriptor for one recording session."""
    session_id: str
    label: str                # "healthy" | "anomaly" | "hardware_healthy" | etc.
    source: str               # "simulated" | "hardware"
    csv_path: str             # absolute path to the CSV file
    start_time: str           # ISO-8601
    end_time: str = ""
    num_rows: int = 0
    metadata: dict = field(default_factory=dict)   # free-form extra info


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Creates, tracks, and finalises sensor data sessions.

    Parameters
    ----------
    base_dir : str | Path
        Root folder for raw data, e.g. ``ml/data/raw/simulated``.
    label : str
        Sub-folder / label for this session type: ``"healthy"`` or ``"anomaly"``.
    source : str
        Data origin tag written to each row and to the manifest.
        One of ``"simulated"`` or ``"hardware"``.
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        base_dir: str | Path,
        label: str = "healthy",
        source: str = "simulated",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.label = label
        self.source = source
        self._label_dir = self.base_dir / label
        self._label_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def new_session(
        self,
        metadata: dict | None = None,
        fieldnames: list[str] | None = None,
    ) -> Generator["_SessionWriter", None, None]:
        """
        Context manager that opens a CSV file for one session.

        yields a ``_SessionWriter`` whose ``.write_row(dict)`` method
        appends one sample row.

        On exit the CSV is flushed, the session info is finalised,
        and the manifest is updated.
        """
        session_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)
        filename = (
            f"{self.source}_{self.label}_{ts.strftime('%Y%m%d_%H%M%S')}_"
            f"{session_id[:8]}.csv"
        )
        csv_path = self._label_dir / filename
        meta_path = csv_path.with_suffix(".meta.json")

        info = SessionInfo(
            session_id=session_id,
            label=self.label,
            source=self.source,
            csv_path=str(csv_path),
            start_time=ts.isoformat(),
            metadata=metadata or {},
        )

        writer = _SessionWriter(
            csv_path=csv_path,
            meta_path=meta_path,
            session_info=info,
            fieldnames=fieldnames,
        )
        try:
            writer._open()
            yield writer
        finally:
            writer._close()
            info.end_time = datetime.now(timezone.utc).isoformat()
            info.num_rows = writer.num_rows
            self._write_session_metadata(info, meta_path)
            self._update_manifest(info)

    def load_manifest(self) -> list[dict]:
        """Return all session records from the manifest."""
        manifest_path = self.base_dir / self.MANIFEST_FILE
        if not manifest_path.exists():
            return []
        with manifest_path.open() as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_manifest(self, info: SessionInfo) -> None:
        manifest_path = self.base_dir / self.MANIFEST_FILE
        records = self.load_manifest()
        records.append(asdict(info))
        with manifest_path.open("w") as f:
            json.dump(records, f, indent=2)

    def _write_session_metadata(self, info: SessionInfo, meta_path: Path) -> None:
        """
        Write a per-session metadata sidecar next to the CSV.
        This makes it easy to move/copy a single session file while keeping context.
        """
        payload = asdict(info)
        # Keep the sidecar small and portable: prefer relative paths when possible.
        try:
            payload["csv_path"] = str(Path(payload["csv_path"]).resolve())
        except Exception:
            pass
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Internal writer
# ---------------------------------------------------------------------------

class _SessionWriter:
    """Low-level CSV writer used inside the SessionManager context manager."""

    def __init__(
        self,
        csv_path: Path,
        meta_path: Path,
        session_info: SessionInfo,
        fieldnames: list[str] | None,
    ) -> None:
        self._csv_path = csv_path
        self._meta_path = meta_path
        self.session_info = session_info
        self._fieldnames = fieldnames
        self._file = None
        self._writer: csv.DictWriter | None = None
        self.num_rows = 0

    def _open(self) -> None:
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        # Header written lazily on first write_row call if fieldnames unknown
        if self._fieldnames:
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

    def write_row(self, row: dict[str, Any]) -> None:
        """Append one sample row.  Infers header from first row if needed."""
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self.num_rows += 1

        # Flush every 50 rows to avoid data loss on interrupt
        if self.num_rows % 50 == 0:
            self._file.flush()

    def _close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
