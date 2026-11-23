from __future__ import annotations

import re
from typing import Iterable

try:
    from ..contracts import NativeLink
except ImportError:
    from contracts import NativeLink

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


def _candidate_spans(markdown: str, hint: str) -> Iterable[str]:
    hint_words = max(1, len(hint.split()))
    words = re.findall(r"\S+", markdown)
    for length in range(hint_words, min(hint_words + 6, len(words) + 1)):
        for start in range(0, len(words) - length + 1):
            yield " ".join(words[start:start + length])


def reinject_links(
    docling_md: str,
    native_links: list[NativeLink],
    docling_json: dict | None = None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    for link in native_links:
        if not link.display_text_hint:
            warnings.append(f"HYPERLINK_NO_TEXT_HINT: {link.uri}")
            continue
        hint = link.display_text_hint
        pattern = re.compile(rf"(?<![\w]){re.escape(hint)}(?![\w])")
        match = pattern.search(docling_md)
        if match:
            context = docling_md[max(0, match.start() - 1):match.end() + len(link.uri) + 5]
            replacement = match.group(0)
            if f"]({link.uri})" not in context:
                replacement = f"[{match.group(0)}]({link.uri})"
            docling_md = docling_md[:match.start()] + replacement + docling_md[match.end():]
            continue
        matched = False
        for candidate in _candidate_spans(docling_md, hint):
            ratio = fuzz.ratio(candidate, hint) if fuzz is not None else 0.0
            if ratio >= 80:
                candidate_match = re.search(rf"(?<![\w]){re.escape(candidate.strip(' .,!?:;'))}(?![\w])", docling_md)
                existing_links = list(re.finditer(r"\[[^\]]+\]\([^)]+\)", docling_md))
                if candidate_match and not any(
                    existing.start() <= candidate_match.start() < existing.end()
                    for existing in existing_links
                ):
                    docling_md = (
                        docling_md[:candidate_match.start()]
                        + f"[{candidate_match.group(0)}]({link.uri})"
                        + docling_md[candidate_match.end():]
                    )
                    warnings.append(f"HYPERLINK_FUZZY_MATCH: {link.uri}")
                    matched = True
                    break
        if not matched:
            warnings.append(f"HYPERLINK_LOST: {link.uri}")
    return docling_md, warnings
