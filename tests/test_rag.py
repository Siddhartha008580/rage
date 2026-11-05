import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.rag import (
    Chunk,
    build_chunks,
    build_index,
    build_lexical_index,
    query_index,
    query_lexical_index,
    select_markdown_files,
)


class RagChunkTests(unittest.TestCase):
    def test_ambiguous_conversion_variants_require_a_ledger_selection(self) -> None:
        markdown = (
            "# Dosage\n\n"
            "| Dose | Frequency |\n"
            "|---|---|\n"
            "| 5 mg | Daily |\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            medical_dir = output_dir / "medical"
            medical_dir.mkdir()
            (medical_dir / "d4--variant-a-v34.md").write_text(
                markdown, encoding="utf-8"
            )
            (medical_dir / "d4--variant-b-v34.md").write_text(
                markdown, encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "does not identify one active output"
            ):
                build_chunks(output_dir)

    def test_ledger_selects_active_conversion_and_skips_stale_variant(self) -> None:
        markdown = (
            "# Dosage\n\n"
            "| Dose | Frequency |\n"
            "|---|---|\n"
            "| 5 mg | Daily |\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            output_dir = workspace / "output" / "markdown"
            medical_dir = output_dir / "medical"
            medical_dir.mkdir(parents=True)
            stale = medical_dir / "d4--variant-a-v34.md"
            active = medical_dir / "d4--variant-b-v34.md"
            stale.write_text(markdown, encoding="utf-8")
            active.write_text(markdown, encoding="utf-8")
            ledger_record = {
                "source": "medical/d4.pdf",
                "status": "converted",
                "markdown": {
                    "status": "converted",
                    "output": "output/markdown/medical/d4--variant-b-v34.md",
                },
            }
            (workspace / "output" / "converted.jsonl").write_text(
                json.dumps(ledger_record) + "\n", encoding="utf-8"
            )

            selected = select_markdown_files(output_dir)
            chunks = build_chunks(output_dir, selected)

        self.assertEqual(selected, [active])
        self.assertEqual({chunk.domain for chunk in chunks}, {"medical"})
        expected_document_id = hashlib.sha256(
            b"medical/d4--variant-b-v34.md"
        ).hexdigest()[:12]
        self.assertTrue(
            all(expected_document_id in chunk.chunk_id for chunk in chunks)
        )

    def test_table_row_chunks_include_headers_full_heading_path_and_domain(self) -> None:
        markdown = (
            "# Finance\n"
            "## Results\n"
            "### Revenue\n"
            "Table 2. Quarterly results\n"
            "| Metric | Q4 |\n"
            "|---|---|\n"
            "| Revenue | 47,310 |\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            medical_dir = output_dir / "finance"
            medical_dir.mkdir()
            path = medical_dir / "report.md"
            path.write_text(markdown, encoding="utf-8")

            chunks = build_chunks(output_dir)

        row_chunk = next(c for c in chunks if c.chunk_type == "table_row")
        self.assertIn("Table 2. Quarterly results", row_chunk.text)
        self.assertIn("Section: Finance > Results > Revenue", row_chunk.text)
        self.assertIn("Columns: Metric | Q4", row_chunk.text)
        self.assertIn("Row: Metric: Revenue | Q4: 47,310", row_chunk.text)
        self.assertEqual(
            row_chunk.metadata()["section_path"],
            ["Finance", "Results", "Revenue"],
        )
        self.assertEqual(row_chunk.metadata()["domain"], "finance")

    def test_heading_path_carries_across_page_boundaries(self) -> None:
        first_page = "# Operations\n## Risk Factors\n### Supply Chain\n\n"
        second_page = (
            "| Risk | Impact |\n"
            "|---|---|\n"
            "| Delay | High |\n"
        )
        markdown = first_page + second_page
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            domain_dir = output_dir / "engineering"
            domain_dir.mkdir()
            path = domain_dir / "report.md"
            path.write_text(markdown, encoding="utf-8")
            path.with_suffix(".pages.json").write_text(
                json.dumps(
                    [
                        {
                            "pdf_page": 1,
                            "char_start": 0,
                            "char_end": len(first_page),
                            "printed_page": "1",
                        },
                        {
                            "pdf_page": 2,
                            "char_start": len(first_page),
                            "char_end": len(markdown),
                            "printed_page": "2",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            chunks = build_chunks(output_dir)

        row_chunk = next(c for c in chunks if c.chunk_type == "table_row")
        self.assertEqual(
            row_chunk.section_path,
            ["Operations", "Risk Factors", "Supply Chain"],
        )

    def test_exact_duplicate_table_chunks_are_emitted_once(self) -> None:
        table = (
            "| Metric | Value |\n"
            "|---|---|\n"
            "| Revenue | 12 |\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            output_dir.mkdir(exist_ok=True)
            (output_dir / "report.md").write_text(
                f"# Results\n{table}\n{table}", encoding="utf-8"
            )

            chunks = build_chunks(output_dir)

        self.assertEqual(
            [chunk.chunk_type for chunk in chunks], ["prose", "table", "table_row"]
        )

    def test_query_combines_domain_page_and_table_filters(self) -> None:
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
        client = MagicMock()
        client.get_collection.return_value = collection
        chromadb = SimpleNamespace(
            PersistentClient=MagicMock(return_value=client)
        )

        with (
            patch.dict(sys.modules, {"chromadb": chromadb}),
            patch("src.rag._embed_fn", return_value=object()),
        ):
            query_index(
                "revenue",
                Path("index"),
                page=3,
                table_only=True,
                domain="financial",
            )

        collection.query.assert_called_once_with(
            query_texts=["revenue"],
            n_results=5,
            where={
                "$and": [
                    {"pdf_page": 3},
                    {"chunk_type": {"$in": ["table", "table_row"]}},
                    {"domain": "financial"},
                ]
            },
        )

    def test_chroma_accepts_chunks_without_heading_paths(self) -> None:
        import chromadb

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "markdown"
            output_dir.mkdir()
            (output_dir / "report.md").write_text(
                "Introductory content without a heading.",
                encoding="utf-8",
            )
            chunks = build_chunks(output_dir)
            self.assertTrue(chunks)
            self.assertTrue(
                all("section_path" not in chunk.metadata() for chunk in chunks)
            )

            client = chromadb.PersistentClient(path=str(Path(directory) / "index"))
            collection = client.create_collection("metadata_regression")
            collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=[[0.0, 1.0] for _ in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata() for chunk in chunks],
            )

            self.assertEqual(collection.count(), len(chunks))

    def test_persistent_bm25_index_returns_ranked_filtered_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            chunks = [
                Chunk(
                    "financial",
                    "quarterly revenue growth",
                    "amzn.pdf",
                    "financial",
                    2,
                    "2",
                    "Results",
                    "",
                    ["Results"],
                    "prose",
                    "",
                    "",
                    "[amzn.pdf p.2]",
                    0,
                    10,
                ),
                Chunk(
                    "medical",
                    "recommended dosage once per day",
                    "d2.pdf",
                    "medical",
                    3,
                    "3",
                    "Dosage",
                    "",
                    ["Dosage"],
                    "table_row",
                    "Dose",
                    "Table 1",
                    "[d2.pdf p.3]",
                    0,
                    10,
                ),
            ]
            build_lexical_index(chunks, index_dir)

            results = query_lexical_index(
                index_dir,
                ["recommended", "dosage"],
                top_k=5,
                page=3,
                table_only=True,
                domain="medical",
            )

        self.assertTrue(results)
        self.assertIn("recommended dosage", results[0]["text"])
        self.assertEqual(results[0]["metadata"]["source"], "d2.pdf")
        self.assertLessEqual(len(results), 5)

    def test_build_index_also_creates_the_lexical_index(self) -> None:
        chunk = Chunk(
            "chunk-1",
            "cash flow statement",
            "report.pdf",
            "financial",
            1,
            "1",
            "Cash Flows",
            "",
            ["Cash Flows"],
            "prose",
            "",
            "",
            "[report.pdf p.1]",
            0,
            20,
        )
        client = MagicMock()
        collection = MagicMock()
        client.create_collection.return_value = collection
        chromadb = SimpleNamespace(
            PersistentClient=MagicMock(return_value=client)
        )
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            with (
                patch.dict(sys.modules, {"chromadb": chromadb}),
                patch("src.rag._embed_fn", return_value=object()),
            ):
                build_index([chunk], index_dir)

            lexical = query_lexical_index(
                index_dir, ["cash", "flow"], top_k=5
            )

        client.create_collection.assert_called_once()
        collection.add.assert_called_once()
        self.assertEqual([hit["id"] for hit in lexical], ["chunk-1"])


if __name__ == "__main__":
    unittest.main()
