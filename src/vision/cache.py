from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


CACHE_DIR = Path(os.environ.get("VISION_CACHE_DIR", str(Path.home() / ".cache" / "vision")))


def cached_analysis(image_path: Path, client: Any, analyzer: Callable[[Path, Any], dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    cache_file = CACHE_DIR / f"{digest}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    analysis = analyzer(image_path, client)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(analysis), encoding="utf-8")
    return analysis
