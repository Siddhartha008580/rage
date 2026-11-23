from __future__ import annotations

import os
import resource
import signal
import subprocess
from pathlib import Path
from typing import Any

try:
    from ..contracts import PageArtifact
except ImportError:
    from contracts import PageArtifact


def _emit_fallback_warning(
    start: int,
    end: int,
    reason: str,
    stderr: str,
    max_log_bytes: int,
) -> dict[str, str]:
    captured = (stderr or "").encode("utf-8")[:max_log_bytes].decode("utf-8", "replace")
    return {"page_range": f"{start}-{end}", "reason": reason, "stderr": captured}


def convert_pages_isolated(
    source: Path,
    page_ranges: list[tuple[int, int]],
    scratch_dir: Path,
    *,
    memory_bytes: int = 4 * 1024**3,
    cpu_seconds: int = 120,
    wall_seconds: int = 300,
    max_log_bytes: int = 1 * 1024**2,
) -> dict[int, PageArtifact]:
    results: dict[int, PageArtifact] = {}
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = scratch_dir / "tmp"
    hf_dir = scratch_dir / "hf"
    tmp_dir.mkdir(exist_ok=True)
    hf_dir.mkdir(exist_ok=True)
    for start, end in page_ranges:
        out_dir = scratch_dir / f"range_{start}_{end}"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _limit() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        cmd = [
            "docling", "convert", str(source),
            "--to", "md", "--to", "json",
            "--image-export-mode", "referenced",
            "--page-range", f"{start}-{end}",
            "--output", str(out_dir),
        ]
        env = {**os.environ, "TMPDIR": str(tmp_dir), "HF_HOME": str(hf_dir)}
        proc = subprocess.Popen(
            cmd,
            preexec_fn=_limit,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
            _emit_fallback_warning(start, end, "DOCLING_TIMEOUT", stderr, max_log_bytes)
            continue
        if proc.returncode != 0:
            _emit_fallback_warning(start, end, "DOCLING_CLI_FAILED", stderr, max_log_bytes)
            continue
        # Output parsing is implemented by the caller; no backend-specific object escapes this seam.
        for page_number in range(start, end + 1):
            results[page_number] = PageArtifact(
                page_number=page_number,
                route="docling",
                markdown="",
                tables=[],
                figures=[],
                native_links=[],
                confidence=0.0,
                reason_codes=["DOCLING_CLI_OUTPUT_PENDING"],
                provenance={"backend": "docling-cli"},
            )
    return results
