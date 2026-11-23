from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class NativeLink:
    rect: tuple[float, float, float, float]
    uri: str | None
    display_text_hint: str | None


@dataclass(frozen=True)
class TableArtifact:
    table_id: str
    source_page: int
    bbox: tuple[float, float, float, float]
    extractor: Literal["native", "docling", "docling+repair", "native+docling_hint"]
    headers: list[str]
    rows: list[list[str | None]]
    raw_html: str
    quality_score: float
    repair_notes: list[str] = field(default_factory=list)
    rejection_reason: str | None = None


@dataclass(frozen=True)
class FigureArtifact:
    figure_id: str
    source_page: int
    caption: str | None
    image_path: Path | None
    association_confidence: float
    enrichment: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PageArtifact:
    page_number: int
    route: Literal["native", "docling"]
    markdown: str
    tables: list[TableArtifact]
    figures: list[FigureArtifact]
    native_links: list[NativeLink]
    confidence: float
    reason_codes: list[str]
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
