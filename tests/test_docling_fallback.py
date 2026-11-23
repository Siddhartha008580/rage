import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.docling_fallback import (
    DOCLING_HEADER,
    _strip_docling_page_artifacts,
    convert_pages_with_docling,
    docling_configuration,
    find_pages_needing_docling,
    page_needs_docling,
    summarize_usage,
)


def _page(
    text: str,
    *,
    words: list[tuple] | None = None,
    blocks: list[dict] | None = None,
    text_blocks: list[tuple] | None = None,
    images: list[dict] | None = None,
) -> SimpleNamespace:
    rect = SimpleNamespace(
        x0=0, y0=0, x1=100, y1=100, width=100, height=100
    )

    def get_text(kind: str = "text"):
        if kind == "words":
            return words or []
        if kind == "dict":
            return {"blocks": blocks or []}
        if kind == "blocks":
            return text_blocks or []
        return text

    return SimpleNamespace(
        rect=rect,
        get_text=get_text,
        get_image_info=lambda: images or [],
    )


class DoclingFallbackTests(unittest.TestCase):
    def test_docling_is_enabled_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(docling_configuration()["enabled"])

    def test_routes_pages_with_separate_text_columns(self) -> None:
        text_blocks = [
            (10, y, 42, y + 12, f"Left paragraph {index}", 0, index, 0)
            for index, y in enumerate((10, 30, 50, 70))
        ] + [
            (60, y, 92, y + 12, f"Right paragraph {index}", 0, index, 0)
            for index, y in enumerate((12, 32, 52, 72))
        ]
        page = _page("x" * 250, text_blocks=text_blocks)

        self.assertEqual(
            page_needs_docling(page),
            (True, "multiple text columns (reading order)"),
        )

    def test_routes_aligned_numeric_columns(self) -> None:
        words = []
        for line_number, value in enumerate(
            ("10", "20", "30", "40", "50")
        ):
            words.extend(
                [
                    (10, 15 + line_number * 10, 50, 20, "Metric", 0, line_number, 0),
                    (
                        80,
                        15 + line_number * 10,
                        98,
                        20,
                        value,
                        0,
                        line_number,
                        1,
                    ),
                ]
            )
        page = _page("x" * 250, words=words)

        self.assertEqual(
            page_needs_docling(page),
            (True, "aligned numeric column (potential table)"),
        )

    def test_routes_pages_with_multiple_dot_leaders(self) -> None:
        blocks = [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": f"Section {i} .... {i + 1}"}]}
                    for i in range(4)
                ],
            }
        ]
        page = _page("x" * 250, blocks=blocks)

        self.assertEqual(
            page_needs_docling(page),
            (True, "dot-leader rows (table of contents)"),
        )

    def test_routes_sparse_full_page_image_with_some_text(self) -> None:
        page = _page(
            "x" * 20,
            images=[{"bbox": (0, 0, 100, 100)}],
        )

        needs_docling, reason = page_needs_docling(page)

        self.assertTrue(needs_docling)
        self.assertIn("full-page image with sparse text", reason)

    def test_finds_flagged_pages_when_fallback_is_enabled(self) -> None:
        page = _page(
            "x" * 20,
            images=[{"bbox": (0, 0, 100, 100)}],
        )
        with patch.dict(
            "os.environ", {"DOCLING_DISABLE": "0"}, clear=False
        ):
            pages = find_pages_needing_docling([page])

        self.assertEqual(list(pages), [0])

    def test_unavailable_docling_records_warning_and_uses_standard_path(self) -> None:
        warnings: list[str] = []
        with patch("src.docling_fallback._check_docling", return_value=False):
            results = convert_pages_with_docling(
                Path("sample.pdf"), {0}, warnings
            )

        self.assertEqual(results, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("Docling is unavailable", warnings[0])

    def test_configuration_reflects_disable_and_ocr_flags(self) -> None:
        with patch.dict(
            "os.environ",
            {"DOCLING_DISABLE": "yes", "DOCLING_OCR": "1"},
            clear=False,
        ):
            self.assertEqual(
                set(docling_configuration()),
                {"enabled", "ocr", "available"},
            )
            configuration = docling_configuration()
            self.assertFalse(configuration["enabled"])
            self.assertTrue(configuration["ocr"])

    def test_docling_settings_change_pdf_output_token(self) -> None:
        from src.ingest import output_token

        source = Path("medical/sample.pdf")
        digest = "a" * 64
        with patch.dict(
            "os.environ",
            {"DOCLING_DISABLE": "1", "DOCLING_OCR": "0"},
            clear=False,
        ):
            disabled_token = output_token(source, digest)
        with patch.dict(
            "os.environ",
            {"DOCLING_DISABLE": "0", "DOCLING_OCR": "1"},
            clear=False,
        ):
            ocr_token = output_token(source, digest)

        self.assertNotEqual(disabled_token, ocr_token)

    def test_strips_docling_page_comments_and_summarizes_routing(self) -> None:
        self.assertEqual(
            _strip_docling_page_artifacts(
                "  <!-- page-break -->\n## Content\n"
            ),
            "## Content",
        )
        self.assertTrue(DOCLING_HEADER.startswith("<!-- rendered-by:"))
        self.assertEqual(
            summarize_usage({0: "dot-leader rows (table of contents)"}),
            "Docling fallback: 1 pages flagged (1x dot-leader rows)",
        )


if __name__ == "__main__":
    unittest.main()
