from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any


MINERU_HEADER = "<!-- rendered-by: mineru -->\n"


def convert_pages_with_mineru(
    source: Path,
    page_indices: set[int],
    warnings: list[str],
    native_links_by_page: dict[int, list[Any]] | None = None,
    *,
    tier: str = "standard",
    timeout_seconds: int = 300,
    max_pages: int = 20,
) -> dict[int, str]:
    """Run MinerU on selected, isolated single-page PDF copies."""
    if not page_indices:
        return {}
    executable = shutil.which("mineru-kit")
    if not executable:
        warnings.append(
            "MinerU fallback requested but mineru-kit is unavailable; "
            "using the standard pipeline"
        )
        return {}
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1")

    selected_pages = sorted(page_indices)
    if len(selected_pages) > max_pages:
        warnings.append(
            f"MinerU fallback limited to {max_pages} pages; "
            f"{len(selected_pages) - max_pages} pages remain on the standard pipeline"
        )
        selected_pages = selected_pages[:max_pages]

    import pymupdf

    results: dict[int, str] = {}
    try:
        from .hyperlink_reinject import reinject_links
    except ImportError:
        from backends.hyperlink_reinject import reinject_links

    with pymupdf.open(source) as document:
        for page_index in selected_pages:
            if page_index < 0 or page_index >= len(document):
                warnings.append(
                    f"MinerU page index {page_index} is outside the PDF; "
                    "using the standard pipeline"
                )
                continue
            with tempfile.TemporaryDirectory(prefix="mineru-page-") as temp_name:
                work_dir = Path(temp_name)
                page_pdf = work_dir / "page.pdf"
                markdown_path = work_dir / "page.md"
                single_page = pymupdf.open()
                try:
                    single_page.insert_pdf(
                        document,
                        from_page=page_index,
                        to_page=page_index,
                    )
                    single_page.save(page_pdf)
                finally:
                    single_page.close()

                command = [
                    executable,
                    "parse",
                    str(page_pdf),
                    "-o",
                    str(markdown_path),
                    "--tier",
                    tier,
                ]
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=work_dir,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    warnings.append(
                        f"MinerU timed out on PDF page {page_index + 1}; "
                        "using the standard pipeline"
                    )
                    continue
                except OSError as error:
                    warnings.append(
                        f"Could not start MinerU on PDF page "
                        f"{page_index + 1}: {error}; using the standard pipeline"
                    )
                    continue

                if process.returncode != 0:
                    warnings.append(
                        f"MinerU failed on PDF page {page_index + 1} "
                        f"(exit {process.returncode}); using the standard "
                        "pipeline"
                    )
                    continue
                try:
                    markdown = markdown_path.read_text(encoding="utf-8").strip()
                except OSError as error:
                    warnings.append(
                        f"MinerU produced no Markdown for PDF page "
                        f"{page_index + 1}: {error}; using the standard pipeline"
                    )
                    continue
                if not markdown:
                    warnings.append(
                        f"MinerU returned empty Markdown for PDF page "
                        f"{page_index + 1}; using the standard pipeline"
                    )
                    continue
                links = (native_links_by_page or {}).get(page_index, [])
                if links:
                    markdown, link_warnings = reinject_links(markdown, links)
                    warnings.extend(
                        f"Page {page_index + 1}: {warning}"
                        for warning in link_warnings
                    )
                results[page_index] = markdown
    return results
