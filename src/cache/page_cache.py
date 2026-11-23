from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from ..contracts import PageArtifact
except ImportError:
    from contracts import PageArtifact


SCHEMA = """
CREATE TABLE IF NOT EXISTS page_cache (
  key TEXT PRIMARY KEY,
  page_number INTEGER NOT NULL,
  document_hash TEXT NOT NULL,
  artifact_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  backend_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document ON page_cache(document_hash);
"""


def page_cache_key(page: Any, route: str, backend_config_hash: str) -> str:
    h = hashlib.sha256()
    h.update(page.get_text().encode())
    for info in page.get_image_info():
        h.update(str(info["bbox"]).encode())
        h.update(str(info.get("size", 0)).encode())
    h.update(route.encode())
    h.update(backend_config_hash.encode())
    return h.hexdigest()


class PageCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def put(self, key: str, page_number: int, document_hash: str, artifact: PageArtifact, created_at: str, backend_version: str) -> None:
        import json

        self.connection.execute(
            "INSERT OR REPLACE INTO page_cache "
            "(key,page_number,document_hash,artifact_json,created_at,backend_version) "
            "VALUES (?,?,?,?,?,?)",
            (key, page_number, document_hash, json.dumps(asdict(artifact)), created_at, backend_version),
        )
        self.connection.commit()
