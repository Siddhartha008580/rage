import unittest

from src.chunk import Block, _child_chunks, parse_markdown
from src.ingest import _render_normalized_table


class TableChunkContextTests(unittest.TestCase):
    def test_normalized_complex_table_is_pipe_markdown_and_remains_a_table_chunk(self) -> None:
        markdown = _render_normalized_table(
            ["Measure", "Value"],
            [
                {
                    "row_group_path": ["Study A"],
                    "row_group_footnote_markers": [],
                    "values": ["Outcome 1", "3.2"],
                    "footnote_markers": [],
                }
            ],
        )

        [block] = parse_markdown(markdown)
        [chunk] = _child_chunks([block], 180, 20)

        self.assertEqual(block.kind, "table")
        self.assertEqual(chunk[1], "table")
        self.assertIn("| Row group | Measure | Value |", chunk[0])
        self.assertIn("Study A", chunk[0])
        self.assertIn("| --- | --- | --- |", chunk[0])
        self.assertIn("| Outcome 1 | 3.2 |", chunk[0])

    def test_html_tables_are_parsed_and_split_with_headers_repeated(self) -> None:
        markdown = "\n".join(
            [
                "**Table 2: Results**",
                "<table>",
                "<thead><tr><th>Measure</th><th>Value</th></tr></thead>",
                "<tbody>",
                "<tr><td>First outcome</td><td>1</td></tr>",
                "<tr><td>Second outcome</td><td>2</td></tr>",
                "<tr><td>Third outcome</td><td>3</td></tr>",
                "</tbody>",
                "</table>",
            ]
        )

        blocks = parse_markdown(markdown)
        table_block = next(block for block in blocks if block.kind == "table")
        chunks = _child_chunks(blocks, 20, 0)

        self.assertGreater(len(chunks), 1)
        for content, kind, *_ in chunks:
            self.assertEqual(kind, "table")
            self.assertTrue(content.startswith("**Table 2: Results**"))
            self.assertIn("<thead><tr><th>Measure</th><th>Value</th></tr></thead>", content)
            self.assertIn("<table>", content)
            self.assertIn("</table>", content)

    def test_caption_is_attached_to_table_and_page_provenance(self) -> None:
        caption = Block("**Table 6: HbA1c results**", 3, "6")
        table = Block(
            "| Measure | Value |\n| --- | --- |\n| HbA1c | 8.2 |",
            4,
            "7",
            kind="table",
        )

        chunks = _child_chunks([caption, table], 180, 20)

        self.assertEqual(len(chunks), 1)
        content, kind, page_start, page_end, printed_start, printed_end = chunks[0]
        self.assertEqual(kind, "table")
        self.assertTrue(content.startswith(caption.text + "\n\n| Measure"))
        self.assertEqual((page_start, page_end), (3, 4))
        self.assertEqual((printed_start, printed_end), ("6", "7"))


if __name__ == "__main__":
    unittest.main()