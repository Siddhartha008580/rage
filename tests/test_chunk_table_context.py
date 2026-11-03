import unittest

from src.chunk import Block, _child_chunks


class TableChunkContextTests(unittest.TestCase):
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