from __future__ import annotations

import re


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        if len(token) >= 3
    }


def vlm_divergence(vlm_text: str, native_text: str) -> tuple[float, list[str]]:
    if len(vlm_text) < 40:
        return 1.0, ["VLM_TOO_SHORT"]
    vlm_tokens = _tokenize(vlm_text)
    native_tokens = _tokenize(native_text)
    if not native_tokens:
        return 0.0, ["NO_NATIVE_BASELINE"]
    novel = vlm_tokens - native_tokens
    ratio = len(novel) / max(1, len(vlm_tokens))
    return ratio, ["VLM_DIVERGENCE"] if ratio > 0.15 else []
