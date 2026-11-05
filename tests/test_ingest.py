import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ingest import (
    _normalized_table_data,
    _pdf_table_caption_from_page,
    _table_header_signature,
    _pdfplumber_tables_by_page,
)


class IngestTableTests(unittest.TestCase):
    def test_context_year_headers_are_added_without_changing_raw_rows(self) -> None:
        words = [
            {"x0": 26, "x1": 34, "top": 5, "bottom": 10, "text": "2024"},
            {"x0": 56, "x1": 64, "top": 5, "bottom": 10, "text": "2025"},
        ]
        table = SimpleNamespace(
            bbox=(0, 20, 80, 50),
            rows=[
                SimpleNamespace(
                    cells=[
                        (0, 20, 20, 35),
                        (20, 20, 50, 35),
                        (50, 20, 80, 35),
                    ]
                ),
                SimpleNamespace(
                    cells=[
                        (0, 35, 20, 50),
                        (20, 35, 50, 50),
                        (50, 35, 80, 50),
                    ]
                ),
            ],
            extract=lambda: [
                ["Revenue", "100", "200"],
                ["Operating costs", "20", "30"],
            ],
        )
        page = SimpleNamespace(
            height=100,
            find_tables=lambda: [table],
            extract_words=lambda: words,
        )
        pdf_context = MagicMock()
        pdf_context.__enter__.return_value = SimpleNamespace(pages=[page])
        pdf_context.__exit__.return_value = False

        with patch("pdfplumber.open", return_value=pdf_context):
            [extracted] = _pdfplumber_tables_by_page(
                Path("sample.pdf"), []
            )[0]

        self.assertEqual(
            extracted["rows"],
            [
                [None, "2024", "2025"],
                ["Revenue", "100", "200"],
                ["Operating costs", "20", "30"],
            ],
        )
        self.assertEqual(
            extracted["raw_rows"],
            [
                ["Revenue", "100", "200"],
                ["Operating costs", "20", "30"],
            ],
        )
        self.assertIn("Column headings recovered", extracted["repair_notes"][0])
        self.assertEqual(
            _table_header_signature(extracted["rows"]),
            ("", "2024", "2025"),
        )

    def test_context_header_does_not_duplicate_headers_in_table_rows(self) -> None:
        rows = [
            ["Metric", "2024", "2025"],
            ["Revenue", "100", "200"],
            ["Costs", "50", "75"],
        ]
        words = [
            {"x0": 26, "x1": 34, "top": 5, "bottom": 10, "text": "2024"},
            {"x0": 56, "x1": 64, "top": 5, "bottom": 10, "text": "2025"},
        ]
        table = SimpleNamespace(
            bbox=(0, 20, 80, 50),
            rows=[
                SimpleNamespace(
                    cells=[
                        (0, 20, 20, 30),
                        (20, 20, 50, 30),
                        (50, 20, 80, 30),
                    ]
                ),
                SimpleNamespace(
                    cells=[
                        (0, 30, 20, 40),
                        (20, 30, 50, 40),
                        (50, 30, 80, 40),
                    ]
                ),
                SimpleNamespace(
                    cells=[
                        (0, 40, 20, 50),
                        (20, 40, 50, 50),
                        (50, 40, 80, 50),
                    ]
                ),
            ],
            extract=lambda: rows,
        )
        page = SimpleNamespace(
            height=100,
            find_tables=lambda: [table],
            extract_words=lambda: words,
        )
        pdf_context = MagicMock()
        pdf_context.__enter__.return_value = SimpleNamespace(pages=[page])
        pdf_context.__exit__.return_value = False

        with patch("pdfplumber.open", return_value=pdf_context):
            [extracted] = _pdfplumber_tables_by_page(
                Path("sample.pdf"), []
            )[0]

        self.assertEqual(extracted["rows"], rows)
        self.assertFalse(extracted.get("repair_notes"))

    def test_caption_detector_rejects_table_of_contents_header(self) -> None:
        words = [
            {"x0": 20, "x1": 40, "top": 10, "bottom": 15, "text": "Table"},
            {"x0": 42, "x1": 50, "top": 10, "bottom": 15, "text": "of"},
            {
                "x0": 52,
                "x1": 90,
                "top": 10,
                "bottom": 15,
                "text": "Contents",
            },
        ]
        page = SimpleNamespace(extract_words=lambda: words)
        table = SimpleNamespace(bbox=(0, 20, 100, 100))

        self.assertIsNone(_pdf_table_caption_from_page(page, table, words))

    def test_total_rows_end_active_row_groups(self) -> None:
        _, rows, _ = _normalized_table_data(
            [
                ["Category", "Value"],
                ["Operating expenses:", None],
                ["Salaries", "10"],
                ["Total operating expenses", "10"],
                ["Other income", "3"],
            ]
        )

        self.assertEqual(rows[0]["row_group_path"], ["Operating expenses"])
        self.assertEqual(rows[1]["row_group_path"], ["Operating expenses"])
        self.assertEqual(rows[1]["values"][0], "Total operating expenses")
        self.assertEqual(rows[2]["row_group_path"], [])


if __name__ == "__main__":
    unittest.main()
