"""Atomic file I/O helpers for index metadata and manifests."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(target_path: Path, content: str) -> None:
    """Write text atomically via a temporary file + fsync + os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_json(target_path: Path, payload: Any) -> None:
    """Serialize JSON atomically so concurrent readers never observe partial files."""
    atomic_write_text(
        target_path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
