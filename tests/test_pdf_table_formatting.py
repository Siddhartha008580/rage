import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ingest import (
    _configured_pdf_table_backend,
    _configured_pdf_table_fallback,
    _insert_pdf_tables,
    _fence_unmarked_pdf_examples,
    _merge_unstructured_fallback,
    _normalize_markdown_tables,
    _normalize_pdf_header,
    _normalize_pdf_fenced_blocks,
    _normalize_pdf_heading_levels,
    _repeated_pdf_furniture,
    _remove_repeated_pdf_furniture,
    _pdf_tables_by_page,
    _pdfplumber_tables_by_page,
    _pdf_table_artifacts,
    _repair_pdf_table_rows,
    _render_normalized_table,
    _table_complexity,
    _table_rows_are_plausible,
    _normalized_table_data,
    _table_to_html,
    _table_to_markdown,
    _unstructured_table_bbox,
)


class PDFTableFormattingTests(unittest.TestCase):
    def test_pdfplumber_is_the_default_pdf_table_backend(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_configured_pdf_table_backend(), "pdfplumber")

    def test_pdfplumber_is_a_selectable_pdf_table_backend(self) -> None:
        with patch.dict("os.environ", {"PDF_TABLE_BACKEND": "pdfplumber"}):
            self.assertEqual(_configured_pdf_table_backend(), "pdfplumber")

    def test_pdfplumber_backend_uses_pdfplumber_extractor(self) -> None:
        expected_tables = {0: [{"rows": [["A", "B"]]}]}
        with patch(
            "src.ingest._pdfplumber_tables_by_page",
            return_value=expected_tables,
        ) as extract:
            result = _pdf_tables_by_page(None, "pdfplumber", [])

        self.assertIs(result, expected_tables)
        extract.assert_called_once_with(None, [])

    def test_sparse_diagram_rows_are_not_classified_as_tables(self) -> None:
        diagram_rows = [
            ["", "\ue000", "", ""],
            ["", "UA ======== A --->", "", ""],
            ["", "\ue001", "", ""],
            ["", "", "", ""],
        ]
        real_table_rows = [
            ["Method", "Description", "Section"],
            ["GET", "Retrieve a representation", "9.3.1"],
            ["PUT", "Replace a representation", "9.3.4"],
        ]

        self.assertFalse(_table_rows_are_plausible(diagram_rows))
        self.assertTrue(_table_rows_are_plausible(real_table_rows))

    def test_merged_table_row_is_reconstructed_from_word_positions(self) -> None:
        page = SimpleNamespace(
            extract_words=lambda: [
                {"x0": 5, "x1": 15, "top": 11, "bottom": 19, "text": "HEAD"},
                {"x0": 25, "x1": 42, "top": 11, "bottom": 19, "text": "Same"},
                {"x0": 44, "x1": 52, "top": 11, "bottom": 19, "text": "as"},
                {"x0": 84, "x1": 96, "top": 11, "bottom": 19, "text": "9.3.2"},
            ]
        )
        table = SimpleNamespace(
            rows=[
                SimpleNamespace(cells=[(0, 0, 20, 10), (20, 0, 80, 10), (80, 0, 100, 10)]),
                SimpleNamespace(cells=[(0, 10, 100, 20), None, None]),
            ]
        )

        repaired, repaired_indexes = _repair_pdf_table_rows(
            page,
            table,
            [
                ["Method", "Description", "Section"],
                ["HEAD Same as 9.3.2", None, None],
            ],
        )

        self.assertEqual(repaired[1], ["HEAD", "Same as", "9.3.2"])
        self.assertEqual(repaired_indexes, [1])

    def test_pdfplumber_uses_context_headers_to_rebuild_merged_rows(self) -> None:
        words = [
            {"x0": 1, "x1": 5, "top": 10, "bottom": 15, "text": "Table"},
            {"x0": 6, "x1": 9, "top": 10, "bottom": 15, "text": "1:"},
            {"x0": 10, "x1": 19, "top": 10, "bottom": 15, "text": "Revenue"},
            {"x0": 26, "x1": 34, "top": 0, "bottom": 4, "text": "2024"},
            {"x0": 56, "x1": 64, "top": 0, "bottom": 4, "text": "2025"},
            {"x0": 3, "x1": 16, "top": 22, "bottom": 26, "text": "Metric"},
            {"x0": 26, "x1": 34, "top": 22, "bottom": 26, "text": "2024"},
            {"x0": 56, "x1": 64, "top": 22, "bottom": 26, "text": "2025"},
            {"x0": 3, "x1": 15, "top": 38, "bottom": 43, "text": "Revenue"},
            {"x0": 26, "x1": 35, "top": 38, "bottom": 43, "text": "100"},
            {"x0": 56, "x1": 65, "top": 38, "bottom": 43, "text": "200"},
            {"x0": 3, "x1": 15, "top": 53, "bottom": 58, "text": "Costs"},
            {"x0": 26, "x1": 35, "top": 53, "bottom": 58, "text": "20"},
            {"x0": 56, "x1": 65, "top": 53, "bottom": 58, "text": "30"},
        ]
        table = SimpleNamespace(
            bbox=(0, 20, 80, 65),
            rows=[
                SimpleNamespace(
                    cells=[(0, 20, 20, 35), (20, 20, 50, 35), (50, 20, 80, 35)]
                ),
                SimpleNamespace(cells=[(0, 35, 80, 50), None, None]),
                SimpleNamespace(cells=[(0, 50, 80, 65), None, None]),
            ],
            extract=lambda: [
                ["Metric", "2024", "2025"],
                ["Revenue 100 200", None, None],
                ["Costs 20 30", None, None],
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
            tables = _pdfplumber_tables_by_page(Path("sample.pdf"), [])

        [extracted] = tables[0]
        self.assertEqual(
            extracted["rows"],
            [
                ["Metric", "2024", "2025"],
                ["Revenue", "100", "200"],
                ["Costs", "20", "30"],
            ],
        )
        self.assertEqual(extracted["raw_rows"][1], ["Revenue 100 200", None, None])
        self.assertEqual(extracted["caption"], "Table 1: Revenue")
        [artifact] = _pdf_table_artifacts({0: [extracted]}, [""])
        self.assertEqual(artifact["caption"], "Table 1: Revenue")

    def test_unstructured_table_fallback_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_configured_pdf_table_fallback(), "none")
        with patch.dict("os.environ", {"PDF_TABLE_FALLBACK": "unstructured"}):
            self.assertEqual(_configured_pdf_table_fallback(), "unstructured")

    def test_unstructured_fills_missing_pdfplumber_tables(self) -> None:
        fallback_table = {
            "bbox": (0, 0, 100, 100),
            "page_height": 100,
            "rows": [["Name", "Value"], ["A", "1"]],
            "html": "<table></table>",
            "table_index": 1,
            "extractor": "unstructured",
        }
        warnings: list[str] = []
        with (
            patch.dict("os.environ", {"PDF_TABLE_FALLBACK": "unstructured"}),
            patch("src.ingest._pdfplumber_tables_by_page", return_value={}),
            patch("src.ingest._unstructured_candidate_pages", return_value={0}),
            patch(
                "src.ingest._unstructured_tables_by_page",
                return_value={0: [fallback_table]},
            ),
        ):
            tables = _pdf_tables_by_page("sample.pdf", "pdfplumber", warnings)

        self.assertEqual(tables[0][0]["extractor"], "unstructured")
        self.assertTrue(any("additional table" in warning for warning in warnings))

    def test_better_unstructured_structure_keeps_pdfplumber_alternative(self) -> None:
        pdfplumber_table = {
            "bbox": (0, 0, 100, 100),
            "rows": [["A", "B"], ["only one cell"]],
            "table_index": 1,
        }
        unstructured_table = {
            "bbox": (0, 0, 100, 100),
            "rows": [["A", "B"], ["one", "two"]],
            "html": "<table></table>",
            "table_index": 1,
            "extractor": "unstructured",
        }
        warnings: list[str] = []

        [selected] = _merge_unstructured_fallback(
            {0: [pdfplumber_table]},
            {0: [unstructured_table]},
            warnings,
        )[0]

        self.assertEqual(selected["extractor"], "unstructured")
        self.assertEqual(selected["alternative"]["rows"], pdfplumber_table["rows"])
        self.assertTrue(any("replaced a structurally weaker" in item for item in warnings))

    def test_unstructured_coordinates_map_back_to_pdf_page(self) -> None:
        coordinates = SimpleNamespace(
            points=[(100, 200), (300, 200), (300, 400), (100, 400)],
            system=SimpleNamespace(
                width=600,
                height=800,
                orientation=SimpleNamespace(value=(1, -1)),
            ),
        )

        bbox = _unstructured_table_bbox(coordinates, 300, 400)

        self.assertEqual(bbox, (50, 100, 150, 200))

    def test_complex_table_normalization_expands_aligned_rows_and_groups(self) -> None:
        source_rows = [
            ["Subject group", "Cmax", "Tmax", "Renal clearance"],
            [
                "Healthy adults:\n500 mg single dose\nd\n"
                "850 mg single dose\ne\n850 mg multiple dose",
                "1.03\n1.60\n2.01",
                "2.75\n2.64\n1.79",
                "600\n552\n642",
            ],
            [
                "f\nElderly adults:\n850 mg single dose",
                "2.45",
                "2.71",
                "412",
            ],
            [
                "Renal-impaired adults:\n850 mg single dose\ng\nMild\n"
                "Moderate\nSevere",
                "1.86\n4.12\n3.93",
                "3.20\n3.75\n4.01",
                "384\n108\n130",
            ],
        ]

        headers, rows, warnings = _normalized_table_data(source_rows)
        markdown = _render_normalized_table(headers, rows)

        self.assertEqual(headers, ["Subject group", "Cmax", "Tmax", "Renal clearance"])
        self.assertEqual(len(rows), 7)
        self.assertTrue(
            all(
                row["row_group_path"] == ["Healthy adults"]
                for row in rows[:3]
            )
        )
        self.assertEqual(rows[1]["values"][1], "1.60")
        self.assertEqual(rows[1]["footnote_markers"], ["d"])
        self.assertEqual(rows[2]["footnote_markers"], ["e"])
        self.assertEqual(rows[3]["row_group_path"], ["Elderly adults"])
        self.assertEqual(rows[3]["row_group_footnote_markers"], ["f"])
        self.assertEqual(
            rows[4]["row_group_path"],
            ["Renal-impaired adults", "850 mg single dose"],
        )
        self.assertEqual(rows[4]["row_group_footnote_markers"], ["g"])
        self.assertEqual(rows[4]["footnote_markers"], [])
        self.assertEqual(warnings, [])
        self.assertTrue(markdown.startswith("| Row group | Subject group |"))
        self.assertIn("| Footnote markers |", markdown)
        self.assertIn("| Healthy adults | 500 mg single dose |", markdown)
        self.assertIn(" | d |", markdown)
        self.assertIn(" | g |", markdown)

    def test_complexity_flags_multiline_and_footnote_structures(self) -> None:
        complexity = _table_complexity(
            [
                ["Measure", "Group A", "Group B"],
                ["Outcome\nBaseline\na", "2\n3", "4\n5"],
            ]
        )

        self.assertTrue(complexity["is_complex"])
        self.assertIn("multiline-cells", complexity["signals"])
        self.assertIn("footnote-markers", complexity["signals"])

    def test_table_sidecar_keeps_raw_rows_and_links_footnotes(self) -> None:
        table = {
            "bbox": (10, 20, 200, 300),
            "page_height": 400,
            "rows": [
                ["Measure", "Value"],
                ["Outcome\nBaseline\na", "2\n3"],
            ],
            "table_index": 1,
        }

        [artifact] = _pdf_table_artifacts(
            {0: [table]},
            ["**Table 7: Example outcome table**\n\n- a Footnote text"],
        )

        self.assertEqual(artifact["table_id"], "page-0001-table-01")
        self.assertEqual(artifact["caption"], "Table 7: Example outcome table")
        self.assertEqual(artifact["raw_rows"], table["rows"])
        self.assertEqual(artifact["footnote_markers"], ["a"])
        self.assertEqual(artifact["footnotes"], {"a": "Footnote text"})
        self.assertEqual(artifact["normalized_rows"][0]["values"], ["Outcome", "2"])
        self.assertEqual(artifact["normalization_status"], "review_required")
        self.assertEqual(artifact["structure_score"], 1.0)

    def test_table_sidecar_retains_rows_before_coordinate_repair(self) -> None:
        original_rows = [["HEAD Same as 9.3.2", None, None]]
        table = {
            "bbox": (10, 20, 200, 300),
            "page_height": 400,
            "rows": [["HEAD", "Same as", "9.3.2"]],
            "raw_rows": original_rows,
            "repair_notes": ["Cells reconstructed from PDF word positions"],
            "table_index": 1,
        }

        [artifact] = _pdf_table_artifacts({0: [table]}, [""])

        self.assertEqual(artifact["raw_rows"], original_rows)
        self.assertEqual(artifact["repair_notes"], table["repair_notes"])
        self.assertEqual(artifact["normalization_status"], "review_required")

    def test_extracted_table_is_not_nested_inside_layout_table(self) -> None:
        markdown = "\n".join(
            [
                "| Left column | Middle column | Right column |",
                "| --- | --- | --- |",
                "| Before | ZZTABLE0001_01ZZ | After |",
                "",
                "Next paragraph.",
            ]
        )
        table_html = "<table>\n<tr><td>Value</td></tr>\n</table>"

        normalized, inserted = _insert_pdf_tables(
            markdown,
            [
                (
                    "ZZTABLE0001_01ZZ",
                    "Table 1 on page 1",
                    table_html,
                )
            ],
        )

        self.assertEqual(inserted, {"ZZTABLE0001_01ZZ"})
        self.assertIn(
            "| Before | **Table 1 on page 1 extracted below.** | After |",
            normalized,
        )
        self.assertLess(
            normalized.index("| Before"),
            normalized.index(table_html),
        )
        self.assertLess(
            normalized.index(table_html),
            normalized.index("Next paragraph."),
        )

    def test_standalone_placeholder_is_replaced_in_place(self) -> None:
        markdown = "Before\n\n**ZZTABLE0002_01ZZ**\n\nAfter"
        table = "| Name | Value |\n| --- | --- |\n| A | 1 |"

        normalized, inserted = _insert_pdf_tables(
            markdown,
            [("ZZTABLE0002_01ZZ", "Table 1 on page 2", table)],
        )

        self.assertEqual(inserted, {"ZZTABLE0002_01ZZ"})
        self.assertEqual(normalized, f"Before\n\n{table}\n\nAfter")

    def test_placeholder_inside_code_fence_is_not_treated_as_an_anchor(self) -> None:
        markdown = "```\nZZTABLE0003_01ZZ\n```"

        normalized, inserted = _insert_pdf_tables(
            markdown,
            [("ZZTABLE0003_01ZZ", "Table 1 on page 3", "| A | B |")],
        )

        self.assertEqual(normalized, markdown)
        self.assertEqual(inserted, set())

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

    def test_unmarked_http_and_abnf_examples_are_fenced(self) -> None:
        markdown = "\n\n".join(
            [
                "GET /index HTTP/1.1 Host: www.example.com Accept-Encoding: gzip",
                "method = token request-target = origin-form / absolute-form",
                "Ordinary explanatory prose remains unchanged.",
                "```text\nGET /example HTTP/1.1\n```",
            ]
        )

        normalized = _fence_unmarked_pdf_examples(markdown)

        self.assertIn(
            "```http\nGET /index HTTP/1.1 Host: www.example.com "
            "Accept-Encoding: gzip\n```",
            normalized,
        )
        self.assertIn(
            "```abnf\nmethod = token\nrequest-target = origin-form / "
            "absolute-form\n```",
            normalized,
        )
        self.assertIn("Ordinary explanatory prose remains unchanged.", normalized)
        self.assertIn("```text\nGET /example HTTP/1.1\n```", normalized)

    def test_numbered_heading_depth_uses_section_number(self) -> None:
        markdown = "\n".join(
            [
                "# **11. Methods**",
                "# **11.3. Challenge and Response**",
                "# **11.3.1. Authentication Scheme**",
            ]
        )

        normalized = _normalize_pdf_heading_levels(markdown, {})

        self.assertIn("# **11. Methods**", normalized)
        self.assertIn("## **11.3. Challenge and Response**", normalized)
        self.assertIn("### **11.3.1. Authentication Scheme**", normalized)

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

    def test_html_table_preserves_headers_cells_and_line_breaks(self) -> None:
        rows = [
            ["Category", "Group <A>", "Group B"],
            [None, "Measure", "Measure"],
            ["Name & notes", "1\n2", "3"],
        ]

        html = _table_to_html(rows)

        self.assertIn("<thead>", html)
        self.assertIn("<th>Group &lt;A&gt;</th>", html)
        self.assertIn("<th>Group B</th>", html)
        self.assertIn("<th>Measure</th>", html)
        self.assertIn("<tbody>", html)
        self.assertIn("<td>Name &amp; notes</td>", html)
        self.assertIn("<td>1<br>2</td>", html)

    def test_html_table_without_detected_header_uses_body_cells(self) -> None:
        html = _table_to_html([["A", "B"], ["C", "D"]])

        self.assertNotIn("<thead>", html)
        self.assertEqual(html.count("<td>"), 4)

    def test_table_normalization_preserves_pipe_format_and_line_breaks(self) -> None:
        markdown = "| Value |\n| --- |\n| Baseline<br/>Change |"

        normalized = _normalize_markdown_tables(markdown)

        self.assertIn("| Value |", normalized)
        self.assertIn("| Baseline<br>Change |", normalized)
        self.assertNotIn("<table>", normalized)

    def test_malformed_pipe_rows_are_repaired_without_touching_code_fences(self) -> None:
        markdown = "\n".join(
            [
                "| Name | Description |",
                "| --- | --- |",
                "| More info: | Errata exist | Datatracker | IPR | Info page |",
                "",
                "```markdown",
                "| Example | Value |",
                "| --- | --- |",
                "| A | 1 |",
                "```",
            ]
        )

        normalized = _normalize_markdown_tables(markdown)

        self.assertIn(
            "| More info: | Errata exist<br>Datatracker<br>IPR<br>Info page |",
            normalized,
        )
        self.assertIn("```markdown\n| Example | Value |", normalized)


if __name__ == "__main__":
    unittest.main()