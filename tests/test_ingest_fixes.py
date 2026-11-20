import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.ingest_fixes import (
    clean_page_markdown,
    consistent_printed_labels,
    plan_source,
    printed_pdf_page_number,
    run_command_streaming,
    value_table_from_rows,
)


class IngestFixesTests(unittest.TestCase):
    def test_worker_command_waits_without_a_timeout(self) -> None:
        process = SimpleNamespace(
            stdout=io.StringIO(
                '__INGEST_RESULT__{"source":"doc.pdf","status":"converted"}\n'
            ),
            wait=Mock(return_value=0),
        )

        with patch("src.ingest_fixes.subprocess.Popen", return_value=process):
            result, return_code = run_command_streaming(
                ["python", "worker.py"], Path(".")
            )

        self.assertEqual(
            result, {"source": "doc.pdf", "status": "converted"}
        )
        self.assertEqual(return_code, 0)
        process.wait.assert_called_once_with()

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

    def test_value_table_keeps_explicit_non_year_column_headers(self) -> None:
        result = value_table_from_rows(
            [
                ["Treatment Arm", "Baseline concentration", "Change (%)"],
                ["Group A", "184", "+13 (7)"],
                ["Group B", "182", "+15 (8)"],
                ["Group C", "130", "+71 (54)"],
            ]
        )

        self.assertIsNone(result)

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
