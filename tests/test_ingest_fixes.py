import unittest
from types import SimpleNamespace

from src.ingest_fixes import (
    clean_page_markdown,
    consistent_printed_labels,
    plan_source,
    printed_pdf_page_number,
    value_table_from_rows,
)


class IngestFixesTests(unittest.TestCase):
    def test_clean_page_removes_timestamp_footer_and_html_wrappers(self) -> None:
        cleaned = clean_page_markdown(
            "10/27/26, 4:55 PM <u>Table of Contents</u>\n"
            "<mark>Highlighted</mark> ~~<u>22</u>~~\n"
        )

        self.assertNotIn("10/27/26", cleaned)
        self.assertEqual(cleaned, "Highlighted 22\n")

    def test_printed_page_detection_rejects_years_and_body_numbers(self) -> None:
        page = SimpleNamespace(
            rect=SimpleNamespace(x0=0, y0=0, width=612, height=792),
            get_text=lambda _kind: {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [
                            {
                                "bbox": (20, 20, 50, 35),
                                "spans": [{"text": "2023"}],
                            },
                            {
                                "bbox": (280, 380, 310, 395),
                                "spans": [{"text": "17"}],
                            },
                            {
                                "bbox": (570, 755, 590, 770),
                                "spans": [{"text": "42"}],
                            },
                        ],
                    }
                ]
            },
        )

        self.assertEqual(printed_pdf_page_number(page), "42")

    def test_printed_labels_reject_large_forward_jumps(self) -> None:
        self.assertEqual(
            consistent_printed_labels(["1", "2", "2023", "4", "5"]),
            ["1", "2", None, "4", "5"],
        )

    def test_value_table_collapses_currency_and_spacer_columns(self) -> None:
        result = value_table_from_rows(
            [
                ["Metric", "", "2024", "", "", "2025"],
                ["Revenue", "$", "1,000", "", "$", "2,000"],
                ["Operating costs", "$", "250", "", "$", "400"],
                ["Net income", "$", "750", "", "$", "1,600"],
            ]
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.records,
            [
                {"label": "Revenue", "values": [1000.0, 2000.0]},
                {"label": "Operating costs", "values": [250.0, 400.0]},
                {"label": "Net income", "values": [750.0, 1600.0]},
            ],
        )
        self.assertIn("| Item | 2024 | 2025 |", result.markdown)

    def test_force_plan_overrides_reuse_and_recovery(self) -> None:
        self.assertEqual(
            plan_source(
                {"source_sha256": "same"},
                "same",
                output_ok=True,
                force=True,
            ),
            "reconvert_forced",
        )


if __name__ == "__main__":
    unittest.main()
