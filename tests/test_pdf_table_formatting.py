import unittest
from types import SimpleNamespace

from src.convert import (
    _normalize_markdown_tables,
    _normalize_pdf_header,
    _normalize_pdf_fenced_blocks,
    _normalize_pdf_heading_levels,
    _repeated_pdf_furniture,
    _remove_repeated_pdf_furniture,
    _table_to_markdown,
)


class PDFTableFormattingTests(unittest.TestCase):
    def test_repeated_edge_furniture_removed_without_removing_body_text(self) -> None:
        pages = []
        for page_number in range(1, 5):
            lines = [
                ("2026-11-03 12:33 PM", 10, 20),
                (f"{page_number}/4", 970, 980),
                ("Opening Balance", 450, 460),
            ]
            page = SimpleNamespace(rect=SimpleNamespace(height=1000))
            page.get_text = lambda _kind, lines=lines: {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [
                            {
                                "bbox": (0, y0, 100, y1),
                                "spans": [{"text": text}],
                            }
                            for text, y0, y1 in lines
                        ],
                    }
                ]
            }
            pages.append(page)

        repeated = _repeated_pdf_furniture(pages)
        markdown = "\n".join(
            [
                "| 2026-11-03 12:33 PM | 1/4 |",
                "| --- | --- |",
                "| Opening Balance | 100 |",
            ]
        )
        cleaned = _remove_repeated_pdf_furniture(markdown, repeated)

        self.assertIn(_normalize_pdf_header("2026-11-03 12:33 PM"), repeated)
        self.assertIn(_normalize_pdf_header("1/4"), repeated)
        self.assertNotIn(_normalize_pdf_header("Opening Balance"), repeated)
        self.assertNotIn("2026-11-03", cleaned)
        self.assertNotIn("| --- | --- |", cleaned)
        self.assertIn("| Opening Balance | 100 |", cleaned)

    def test_fenced_prose_is_unwrapped_but_structured_code_is_preserved(self) -> None:
        markdown = "\n".join(
            [
                "```",
                "   The organization reported steady growth across all regions.",
                "   Operating costs also declined during the reporting period.",
                "```",
                "```",
                "struct {",
                "    uint16 count;",
                "};",
                "```",
            ]
        )

        normalized = _normalize_pdf_fenced_blocks(markdown)

        self.assertIn(
            "The organization reported steady growth across all regions. "
            "Operating costs also declined during the reporting period.",
            normalized,
        )
        self.assertIn("```\nstruct {\n    uint16 count;\n};\n```", normalized)
        self.assertEqual(normalized.count("```"), 2)

    def test_aligned_multicolumn_text_remains_fenced(self) -> None:
        markdown = "\n".join(
            [
                "```",
                "Organization: Example Holdings             Report date: 2025",
                "Prepared by: Sample Group                    Version: 2",
                "Department: Research and Analysis            Region: North",
                "```",
            ]
        )

        normalized = _normalize_pdf_fenced_blocks(markdown)

        self.assertIn("```\nOrganization: Example Holdings", normalized)
        self.assertIn("Report date: 2025", normalized)

    def test_numbered_headings_use_outline_depth(self) -> None:
        markdown = "\n".join(
            [
                "```",
                "1. Revenue",
                "```",
                "```",
                "1.2. Regional Results",
                "```",
            ]
        )

        normalized = _normalize_pdf_fenced_blocks(markdown)

        self.assertIn("# 1. Revenue", normalized)
        self.assertIn("## 1.2. Regional Results", normalized)
        self.assertNotIn("```", normalized)

    def test_pdf_heading_hierarchy_uses_case_and_source_font_sizes(self) -> None:
        markdown = "\n".join(
            [
                "# **ACME Corp.**",
                "## **DESCRIPTION**",
                "# **FINANCIAL RESULTS**",
                "# **Revenue by Segment**",
                "# **Operating Metrics**",
                "## **Adjusted EBITDA**",
                "# **Table 1: Regional Revenue**",
            ]
        )
        font_sizes = {
            "description": 14,
            "financial results": 14,
            "revenue by segment": 14,
            "operating metrics": 14,
            "adjusted ebitda": 12,
        }

        normalized = _normalize_pdf_heading_levels(markdown, font_sizes)

        self.assertIn("# **ACME Corp.**", normalized)
        self.assertIn("# **DESCRIPTION**", normalized)
        self.assertIn("# **FINANCIAL RESULTS**", normalized)
        self.assertIn("## **Revenue by Segment**", normalized)
        self.assertIn("## **Operating Metrics**", normalized)
        self.assertIn("### **Adjusted EBITDA**", normalized)
        self.assertIn("**Table 1: Regional Revenue**", normalized)

    def test_all_caps_headings_have_no_brand_specific_exception(self) -> None:
        markdown = "\n".join(
            [
                "# **FINANCIAL RESULTS**",
                "# **ACME CORP.**<sup>**®**</sup>",
                "# **How did revenue change?**",
                "# **Table 2: Revenue by Region**",
            ]
        )
        font_sizes = {
            "financial results": 14,
            "acme corp.®": 18,
            "how did revenue change?": 14,
        }

        normalized = _normalize_pdf_heading_levels(markdown, font_sizes)

        self.assertIn("# **ACME CORP.**<sup>**®**</sup>", normalized)
        self.assertIn("## **How did revenue change?**", normalized)
        self.assertIn("**Table 2: Revenue by Region**", normalized)

    def test_blank_stub_header_uses_extracted_column_names(self) -> None:
        rows = [
            ["", "Region A\nNorth", "Region B\nSouth", "Total"],
            [
                "Revenue\nBaseline\nAnnual change",
                "241.5\n–53.0",
                "237.7\n6.3",
                "478.0\n–46.7",
            ],
        ]

        markdown = _table_to_markdown(rows)

        self.assertIn(
            "|  | Region A<br>North | Region B<br>South | Total |",
            markdown,
        )
        self.assertNotIn("Column 1", markdown)

    def test_multiline_cells_preserve_subrow_alignment(self) -> None:
        rows = [
            ["", "Revenue Summary", None, "Prior Year"],
            [None, "North Region", "South Region", None],
            [
                "Total Revenue\nBaseline\nChange at Year End\nRate",
                "120.0\n132.0\n12.0\n10%",
                "140.0\n150.0\n10.0\n7%",
                "260.0\n282.0\n22.0\n8%",
            ],
        ]

        markdown = _table_to_markdown(rows)

        self.assertIn(
            "|  | Revenue Summary<br>North Region "
            "| Revenue Summary<br>South Region | Prior Year |",
            markdown,
        )
        self.assertIn("North Region", markdown)
        self.assertIn(
            "Total Revenue<br>Baseline<br>Change at Year End<br>Rate",
            markdown,
        )
        self.assertIn("120.0<br>132.0<br>12.0<br>10%", markdown)
        self.assertNotIn("Prior Year<br>South Region", markdown)
        self.assertNotIn("Column 1", markdown)

    def test_explicit_header_terms_are_not_replaced_with_generic_columns(self) -> None:
        rows = [
            ["Category", "Group A", "Group B"],
            [None, "North", "South"],
            ["Revenue", "53.2", "11.7"],
        ]

        markdown = _table_to_markdown(rows)

        self.assertTrue(markdown.startswith("| Category |"))
        self.assertNotIn("Column 1", markdown)
        self.assertIn("Group A<br>North", markdown)

    def test_sparse_multilevel_header_detected_without_domain_vocabulary(self) -> None:
        rows = [
            ["Measure", "Group A (n=80)", "Group B (n=90)"],
            [None, "Reported value", None],
            ["Outcome A", "9.6", "2.6"],
            ["Outcome B", "6.5", "1.5"],
        ]

        markdown = _table_to_markdown(rows)

        self.assertIn(
            "| Measure | Group A (n=80)<br>Reported value | Group B (n=90) |",
            markdown,
        )
        self.assertNotIn("Column 1", markdown)

    def test_cell_delimiters_are_escaped_without_losing_line_breaks(self) -> None:
        rows = [["Name", "Value"], ["A | B\ncontinued", "1\n2"]]

        markdown = _table_to_markdown(rows)

        self.assertIn(r"A \| B<br>continued", markdown)
        self.assertIn("1<br>2", markdown)

    def test_table_normalization_preserves_cell_line_breaks(self) -> None:
        markdown = "| Value |\n| --- |\n| Baseline<br/>Change |"

        normalized = _normalize_markdown_tables(markdown)

        self.assertIn("Baseline<br>Change", normalized)


if __name__ == "__main__":
    unittest.main()