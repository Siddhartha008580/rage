from __future__ import annotations

from typing import Any

try:
    from ..contracts import NativeLink
except ImportError:
    from contracts import NativeLink


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, ...]) -> bool:
    x_overlap = max(0.0, min(a[2], float(b[2])) - max(a[0], float(b[0])))
    y_overlap = max(0.0, min(a[3], float(b[3])) - max(a[1], float(b[1])))
    return x_overlap > 0 and y_overlap > 0


def extract_native_links(page: Any) -> list[NativeLink]:
    links: list[NativeLink] = []
    words = page.get_text("words")
    for link in page.get_links():
        if link.get("kind") != 2 or not link.get("uri"):
            continue
        rect = tuple(float(value) for value in link["from"])
        overlapping = [
            str(word[4]) for word in words
            if _rects_overlap(rect, tuple(word[:4]))
        ]
        links.append(NativeLink(
            rect=rect,
            uri=link["uri"],
            display_text_hint=" ".join(overlapping) or None,
        ))
    return links
