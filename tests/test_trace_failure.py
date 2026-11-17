import json
import tempfile
import unittest
from pathlib import Path

from scripts.trace_failure import _markdown_by_source, trace_one


class TraceFailureTests(unittest.TestCase):
    def _markdown(self, directory: Path, text: str) -> dict[str, list[Path]]:
        markdown_dir = directory / "medical"
        markdown_dir.mkdir()
        markdown_path = markdown_dir / "d1--active.md"
        markdown_path.write_text(text, encoding="utf-8")
        (markdown_dir / "d1--active.tables.json").write_text(
            json.dumps({"source_file": "d1.pdf"}), encoding="utf-8"
        )
        return _markdown_by_source(directory)

    def test_terms_can_be_split_across_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            markdown = self._markdown(Path(temp_dir), "renal dosing")
            spec = {
                "id": "M-08",
                "expected_sources": ["d1.pdf"],
                "expected_terms": ["renal", "dosing"],
                "expected_pages": [],
            }
            chunks = [
                {"source": "d1.pdf", "text": "renal clearance", "pdf_page": 3},
                {"source": "d1.pdf", "text": "dosing guidance", "pdf_page": 26},
            ]

            result = trace_one(spec, chunks, markdown)

        self.assertEqual(result["verdict"], "RETRIEVAL_LOSS")
        self.assertEqual(result["term_chunk_pages"]["renal"]["d1.pdf"], [3])
        self.assertEqual(result["term_chunk_pages"]["dosing"]["d1.pdf"], [26])

    def test_missing_expected_page_is_review_not_assumed_eval_bug(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            markdown = self._markdown(Path(temp_dir), "500 mg")
            spec = {
                "id": "S-06",
                "expected_sources": ["d1.pdf"],
                "expected_terms": ["500 mg"],
                "expected_pages": [24],
            }
            chunks = [{"source": "d1.pdf", "text": "500 mg", "pdf_page": 25}]

            result = trace_one(spec, chunks, markdown)

        self.assertEqual(result["verdict"], "PAGE_LABEL_REVIEW")
        self.assertNotEqual(result["verdict"], "EVAL_BUG")

    def test_term_present_in_markdown_but_absent_from_chunks_is_chunker_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            markdown = self._markdown(Path(temp_dir), "stock-based compensation")
            spec = {
                "id": "M-03",
                "expected_sources": ["d1.pdf"],
                "expected_terms": ["stock-based compensation"],
                "expected_pages": [],
            }

            result = trace_one(spec, [], markdown)

        self.assertEqual(result["verdict"], "CHUNKER_LOSS")


if __name__ == "__main__":
    unittest.main()
