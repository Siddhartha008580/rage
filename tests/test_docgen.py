import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.docgen import (
    _call_llm,
    _extract_json,
    _pdf_safe_text,
    _validate_outline,
    assemble,
    build_outline,
    generate_with_markdown,
    gather_section_sources,
    write_section,
)


class DocgenTests(unittest.TestCase):
    def test_generate_with_markdown_uses_requested_output_directory(self) -> None:
        output_dir = Path("custom-documents")
        with (
            patch("src.docgen.generate_markdown", return_value="# Brief"),
            patch("src.docgen.markdown_to_pdf") as render_pdf,
        ):
            path, markdown = generate_with_markdown(
                "Write a brief", output_dir=output_dir
            )

        self.assertEqual(path.parent, output_dir)
        self.assertEqual(path.suffix, ".pdf")
        self.assertEqual(markdown, "# Brief")
        render_pdf.assert_called_once_with("# Brief", path)

    def test_call_llm_does_not_retry_rate_limit_as_reasoning_fallback(self) -> None:
        class RateLimitError(Exception):
            status_code = 429

        client = MagicMock()
        client.chat.completions.create.side_effect = RateLimitError(
            "429 Too Many Requests"
        )

        with self.assertRaisesRegex(RateLimitError, "Too Many Requests"):
            _call_llm(client, "system", "user")

        client.chat.completions.create.assert_called_once()

    def test_extract_json_reads_object_after_surrounding_text(self) -> None:
        parsed = _extract_json(
            'Here is the outline: {"title":"Brief","sections":[]} Done.'
        )

        self.assertEqual(parsed["title"], "Brief")

    def test_pdf_text_keeps_regulatory_symbols_readable(self) -> None:
        self.assertEqual(
            _pdf_safe_text("β-carotene ≤ 0.2′ and π"),
            "beta-carotene <= 0.2' and \\u03c0",
        )

    def test_build_outline_retries_incomplete_json(self) -> None:
        valid_response = (
            '{"title":"Interaction brief","sections":[{"heading":"Data",'
            '"purpose":"Summarize the data.","queries":["ketoconazole oxycodone"]}]}'
        )
        with patch(
            "src.docgen._call_llm",
            side_effect=[
                ('{"title":"Interaction brief","sections":[{"purpose":', "model", "length"),
                (valid_response, "model", "stop"),
            ],
        ) as call_llm:
            outline, model = build_outline(object(), "Write a brief", 2)

        self.assertEqual(outline["title"], "Interaction brief")
        self.assertEqual(model, "model")
        self.assertEqual(call_llm.call_count, 2)
        self.assertEqual(call_llm.call_args_list[0].kwargs["max_tokens"], 2500)
        self.assertEqual(call_llm.call_args_list[1].kwargs["max_tokens"], 4096)

    def test_write_section_retries_invalid_citations_once(self) -> None:
        with (
            patch(
                "src.docgen._call_llm",
                return_value=("A factual claim without a citation.", "model", "stop"),
            ) as call_llm,
            patch(
                "src.docgen._find_blocked_negative_claims",
                return_value=([], []),
            ),
            patch("src.docgen._find_false_deferral_attributions", return_value=[]),
        ):
            result = write_section(
                object(),
                "Write a brief",
                {"heading": "Summary", "purpose": "Summarize the facts."},
                [
                    {
                        "id": "chunk-1",
                        "citation": "[source.pdf p.1]",
                        "source": "source.pdf",
                        "page": 1,
                        "chunk_type": "prose",
                        "text": "Fact",
                    }
                ],
                index_dir=None,
            )

        self.assertEqual(result["status"], "ungrounded")
        self.assertEqual(call_llm.call_count, 2)

    def test_write_section_retries_truncation_and_fails_closed(self) -> None:
        chunk = {
            "id": "chunk-1",
            "citation": "[source.pdf p.1]",
            "source": "source.pdf",
            "page": 1,
            "chunk_type": "prose",
            "text": "Fact",
        }
        with (
            patch(
                "src.docgen._call_llm",
                side_effect=[
                    ("Partial [source.pdf p.1]", "model", "length"),
                    ("Still partial [source.pdf p.1]", "model", "length"),
                ],
            ) as call_llm,
            patch(
                "src.docgen._find_blocked_negative_claims",
                return_value=([], []),
            ),
            patch("src.docgen._find_false_deferral_attributions", return_value=[]),
        ):
            result = write_section(
                object(),
                "Write a brief",
                {"heading": "Summary", "purpose": "Summarize the facts."},
                [chunk],
                index_dir=None,
            )

        self.assertEqual(result["status"], "truncated")
        self.assertEqual(call_llm.call_count, 2)
        self.assertEqual(
            call_llm.call_args_list[1].kwargs["max_tokens"], 2600
        )

    def test_section_retrieval_reserves_chunks_for_each_outline_query(self) -> None:
        def hits(prefix: str) -> list[dict[str, object]]:
            return [
                {
                    "id": f"{prefix}-{index}",
                    "text": f"{prefix} evidence {index}",
                    "citation": f"[{prefix}.pdf p.{index + 1}]",
                    "source": f"{prefix}.pdf",
                    "page": index + 1,
                    "chunk_type": "prose",
                    "rerank_score": 1.0 - index / 100,
                }
                for index in range(20)
            ]

        with (
            patch(
                "src.docgen.retrieve_chunks",
                side_effect=[hits("first"), hits("second")],
            ) as retrieve,
            patch("src.docgen._lexical_section_sweep", return_value=[]),
        ):
            chunks = gather_section_sources(
                {
                    "heading": "Key provisions",
                    "purpose": "Cover all aspects.",
                    "queries": ["first topic", "second topic"],
                },
                index_dir=MagicMock(),
            )

        self.assertEqual(len(chunks), 14)
        self.assertTrue(any(chunk["id"].startswith("first-") for chunk in chunks))
        self.assertTrue(any(chunk["id"].startswith("second-") for chunk in chunks))
        self.assertTrue(
            all(call.kwargs["top_k"] == 40 for call in retrieve.call_args_list)
        )

    def test_outline_adds_broad_query_after_specific_queries(self) -> None:
        outline = _validate_outline(
            {
                "title": "Label briefing",
                "sections": [
                    {
                        "heading": "Retinopathy warnings",
                        "purpose": "Summarize warning language.",
                        "queries": [
                            "semaglutide warning",
                            "OZEMPIC retinopathy",
                            "RYBELSUS retinopathy",
                        ],
                    }
                ],
            },
            max_sections=2,
            request="Compare retinopathy warnings across GLP-1 labels",
        )

        self.assertEqual(len(outline["sections"][0]["queries"]), 4)
        self.assertEqual(
            outline["sections"][0]["queries"][-1],
            "Retinopathy warnings Compare retinopathy warnings across GLP-1 labels",
        )

    def test_assemble_omits_ungrounded_body_and_cleans_brand_names(self) -> None:
        markdown = assemble(
            "Write a label briefing",
            {"title": "Label briefing"},
            [
                {
                    "heading": "Warnings",
                    "status": "ok",
                    "content": "RYBELSUs(R) warnings are described [d3.pdf p.3].",
                    "citations": ["[d3.pdf p.3]"],
                },
                {
                    "heading": "Unsupported mechanism",
                    "status": "ungrounded",
                    "content": "_Generation failed to produce valid source citations._",
                    "citations": [],
                },
            ],
            "test-model",
        )

        self.assertIn("RYBELSUS® warnings", markdown)
        self.assertIn("Unsupported mechanism ⚠", markdown)
        self.assertNotIn("## Unsupported mechanism", markdown)
        self.assertNotIn("Generation failed to produce valid source citations", markdown)
        self.assertIn("1/2 sections fully grounded", markdown)

    def test_assemble_marks_redundant_section_as_omitted(self) -> None:
        markdown = assemble(
            "Write a label briefing",
            {"title": "Label briefing"},
            [
                {
                    "heading": "Warnings",
                    "status": "ok",
                    "content": "Warning details [d3.pdf p.3].",
                    "citations": ["[d3.pdf p.3]"],
                },
                {
                    "heading": "Related warnings",
                    "status": "redundant",
                    "content": "",
                    "citations": [],
                },
            ],
            "test-model",
        )

        self.assertIn("Related warnings (redundant; omitted)", markdown)
        self.assertNotIn("## Related warnings", markdown)

    def test_assemble_omits_truncated_section_body(self) -> None:
        markdown = assemble(
            "Write a briefing",
            {"title": "Briefing"},
            [
                {
                    "heading": "Detailed provisions",
                    "status": "truncated",
                    "content": "This paragraph was cut off",
                    "citations": [],
                }
            ],
            "test-model",
        )

        self.assertIn("Detailed provisions (truncated; omitted)", markdown)
        self.assertNotIn("## Detailed provisions", markdown)
        self.assertNotIn("This paragraph was cut off", markdown)


if __name__ == "__main__":
    unittest.main()
