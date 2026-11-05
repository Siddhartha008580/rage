import unittest
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, call, patch

from src.chat_rag import (
    SYSTEM_PROMPT,
    ask_openrouter,
    build_context,
    diversify_candidates,
    is_comparison_query,
    rerank_candidates,
    reciprocal_rank_fusion,
    retrieve_chunks,
    run_interactive,
    select_best_per_source,
)


class ChatRagTests(unittest.TestCase):
    def test_retrieval_passes_domain_table_and_page_filters(self) -> None:
        metadata = {
            "source": "RFC9110.pdf",
            "domain": "standards",
            "pdf_page": 3,
            "citation": "[RFC9110.pdf p.3]",
            "chunk_type": "table_row",
            "section": "Fields",
            "section_path": ["HTTP", "Fields"],
        }
        dense_hits = [
            {
                "id": "chunk-1",
                "text": "[RFC9110.pdf p.3]\n| Header | Value |",
                "metadata": metadata,
                "distance": 0.1,
            },
            {
                "id": "chunk-2",
                "text": "[RFC9110.pdf p.3]\n| Other | Value |",
                "metadata": {**metadata, "chunk_type": "table"},
                "distance": 0.2,
            },
        ]
        lexical_hits = [
            {
                "id": "chunk-2",
                "text": dense_hits[1]["text"],
                "metadata": dense_hits[1]["metadata"],
                "bm25_score": -0.5,
            },
            {
                "id": "chunk-1",
                "text": dense_hits[0]["text"],
                "metadata": metadata,
                "bm25_score": -0.4,
            },
        ]
        reranker = type(
            "FakeCrossEncoder",
            (),
            {"predict": lambda self, pairs, show_progress_bar: [0.9, 0.1]},
        )()
        with (
            patch("src.chat_rag.query_index", return_value=dense_hits) as query,
            patch(
                "src.chat_rag.query_lexical_index",
                return_value=lexical_hits,
            ) as lexical_query,
            patch("src.chat_rag._reranker", return_value=reranker),
        ):
            chunks = retrieve_chunks(
                "header value",
                1,
                index_dir=Path("test-index"),
                domain="standards",
                table_only=True,
                page=3,
                candidate_k=2,
            )

        query.assert_called_once_with(
            "header value",
            Path("test-index"),
            top_k=2,
            page=3,
            table_only=True,
            domain="standards",
        )
        lexical_query.assert_called_once_with(
            Path("test-index"),
            ["header", "value"],
            top_k=2,
            page=3,
            table_only=True,
            domain="standards",
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["id"], "chunk-1")
        self.assertEqual(chunks[0]["section_path"], ["HTTP", "Fields"])
        self.assertIn("rrf_score", chunks[0])
        self.assertIn("bm25_rank", chunks[0])
        self.assertIn("rerank_score", chunks[0])

    def test_rrf_fuses_dense_and_lexical_rankings(self) -> None:
        dense = [
            {"id": "a", "text": "A"},
            {"id": "b", "text": "B"},
        ]
        lexical = [
            {"id": "b", "text": "B"},
            {"id": "c", "text": "C"},
        ]
        fused = reciprocal_rank_fusion(
            dense, lexical, rrf_k=60, candidate_k=3
        )
        self.assertEqual([hit["id"] for hit in fused], ["b", "a", "c"])
        self.assertEqual(fused[0]["dense_rank"], 2)
        self.assertEqual(fused[0]["bm25_rank"], 1)

    def test_cross_encoder_reranks_by_predicted_relevance(self) -> None:
        candidates = [
            {"id": "a", "text": "first candidate"},
            {"id": "b", "text": "second candidate"},
        ]
        model = type(
            "FakeCrossEncoder",
            (),
            {"predict": lambda self, pairs, show_progress_bar: [0.2, 0.9]},
        )()
        with patch("src.chat_rag._reranker", return_value=model):
            reranked = rerank_candidates("query", candidates)
        self.assertEqual([hit["id"] for hit in reranked], ["b", "a"])
        self.assertEqual(reranked[0]["rerank_score"], 0.9)

    def test_mmr_diversifies_similar_chunks_and_keeps_best_at_end(self) -> None:
        candidates = [
            {
                "id": "best",
                "text": "revenue growth sales report",
                "rerank_score": 0.95,
            },
            {
                "id": "near-duplicate",
                "text": "revenue growth sales report yearly",
                "rerank_score": 0.9,
            },
            {
                "id": "different",
                "text": "risk factors supply chain disruption",
                "rerank_score": 0.7,
            },
        ]
        selected = diversify_candidates(candidates, 2, mmr_lambda=0.7)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[-1]["id"], "best")
        self.assertNotEqual(selected[0]["id"], "near-duplicate")

    def test_mmr_keeps_distinct_rows_from_the_same_table(self) -> None:
        metadata = {
            "chunk_type": "table_row",
            "source": "report.pdf",
            "pdf_page": 4,
            "char_start": 100,
            "char_end": 200,
        }
        candidates = [
            {
                "id": "row-a",
                "text": "Revenue table Columns Metric Year Row Revenue 100",
                "rerank_score": 0.9,
                "metadata": metadata,
            },
            {
                "id": "row-b",
                "text": "Revenue table Columns Metric Year Row Costs 80",
                "rerank_score": 0.8,
                "metadata": metadata,
            },
            {
                "id": "other",
                "text": "Revenue table Columns Metric Year Row Revenue 100",
                "rerank_score": 0.7,
                "metadata": {
                    **metadata,
                    "source": "other.pdf",
                },
            },
        ]

        selected = diversify_candidates(candidates, 2, mmr_lambda=0.7)

        self.assertEqual({hit["id"] for hit in selected}, {"row-a", "row-b"})

    def test_comparison_selection_covers_sources_before_filling(self) -> None:
        candidates = [
            {
                "id": "a-best",
                "text": "parallel warning text",
                "rerank_score": 0.95,
                "metadata": {"source": "a.pdf"},
            },
            {
                "id": "a-next",
                "text": "another section from a",
                "rerank_score": 0.9,
                "metadata": {"source": "a.pdf"},
            },
            {
                "id": "b-best",
                "text": "parallel warning text",
                "rerank_score": 0.85,
                "metadata": {"source": "b.pdf"},
            },
        ]

        selected = select_best_per_source(candidates, 3)

        self.assertEqual(
            {hit["id"] for hit in selected},
            {"a-best", "a-next", "b-best"},
        )
        self.assertEqual(
            [hit["id"] for hit in selected],
            ["b-best", "a-next", "a-best"],
        )
        self.assertTrue(is_comparison_query("Compare warnings across labels"))
        self.assertTrue(is_comparison_query("What are the differences?"))
        self.assertFalse(is_comparison_query("What is the recommended dose?"))

    def test_retrieval_uses_source_coverage_for_comparison_queries(self) -> None:
        reranked = [
            {
                "id": "a-best",
                "text": "parallel warning text",
                "metadata": {
                    "source": "a.pdf",
                    "pdf_page": 1,
                    "citation": "[a.pdf p.1]",
                    "chunk_type": "prose",
                },
                "rrf_score": 0.1,
                "dense_rank": 1,
                "bm25_rank": 1,
                "rerank_score": 0.95,
            },
            {
                "id": "a-next",
                "text": "another section from a",
                "metadata": {
                    "source": "a.pdf",
                    "pdf_page": 2,
                    "citation": "[a.pdf p.2]",
                    "chunk_type": "prose",
                },
                "rrf_score": 0.08,
                "dense_rank": 2,
                "bm25_rank": 2,
                "rerank_score": 0.9,
            },
            {
                "id": "b-best",
                "text": "parallel warning text",
                "metadata": {
                    "source": "b.pdf",
                    "pdf_page": 1,
                    "citation": "[b.pdf p.1]",
                    "chunk_type": "prose",
                },
                "rrf_score": 0.07,
                "dense_rank": 3,
                "bm25_rank": 3,
                "rerank_score": 0.85,
            },
        ]
        with (
            patch("src.chat_rag.query_index", return_value=[]),
            patch("src.chat_rag.query_lexical_index", return_value=[]),
            patch("src.chat_rag.rerank_candidates", return_value=reranked),
        ):
            chunks = retrieve_chunks(
                "compare warnings across labels",
                2,
                candidate_k=2,
            )

        self.assertEqual({chunk["source"] for chunk in chunks}, {"a.pdf", "b.pdf"})
        self.assertEqual({chunk["id"] for chunk in chunks}, {"a-best", "b-best"})

    def test_openrouter_request_has_larger_budget_and_returns_diagnostics(self) -> None:
        create = Mock(
            return_value=SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=" Answer "),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=20,
                    completion_tokens=5,
                    total_tokens=25,
                ),
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        openai_module = ModuleType("openai")
        openai_module.OpenAI = Mock(return_value=client)
        chunks = [
            {
                "id": "chunk-1",
                "text": "Fact.",
                "source": "a.pdf",
                "domain": "",
                "page": 1,
                "citation": "[a.pdf p.1]",
                "chunk_type": "prose",
                "section_path": [],
            }
        ]

        with (
            patch.dict("sys.modules", {"openai": openai_module}),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        ):
            result = ask_openrouter("question", chunks)

        self.assertEqual(create.call_args.kwargs["max_tokens"], 2000)
        self.assertIn("each relevant source", SYSTEM_PROMPT)
        self.assertEqual(result["answer"], "Answer")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(
            result["usage"],
            {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        )

    def test_context_uses_general_source_page_citation(self) -> None:
        context = build_context(
            [
                {
                    "id": "chunk-1",
                    "text": "[RFC9110.pdf p.3]\nHeader information.",
                    "source": "RFC9110.pdf",
                    "domain": "standards",
                    "page": 3,
                    "citation": "[RFC9110.pdf p.3]",
                    "chunk_type": "prose",
                    "section": "Fields",
                    "section_path": ["HTTP", "Fields"],
                }
            ]
        )

        self.assertIn("citation=[RFC9110.pdf p.3]", context)
        self.assertIn("section_path=HTTP > Fields", context)
        self.assertIn("exact source-page citation", context)
        self.assertNotIn("[dN.pdf", context)

    def test_interactive_cli_answers_multiple_queries_and_exits(self) -> None:
        with (
            patch("builtins.input", side_effect=["first question", "second", "quit"]),
            patch("src.chat_rag.run_one", return_value={"answer": "ok"}) as run,
            patch("builtins.print"),
        ):
            run_interactive(
                top_k=3,
                index_dir=Path("test-index"),
                domain="standards",
                table_only=True,
                candidate_k=20,
            )

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    "first question",
                    False,
                    top_k=3,
                    index_dir=Path("test-index"),
                    domain="standards",
                    table_only=True,
                    page=None,
                    candidate_k=20,
                    mmr_lambda=0.7,
                ),
                call(
                    "second",
                    False,
                    top_k=3,
                    index_dir=Path("test-index"),
                    domain="standards",
                    table_only=True,
                    page=None,
                    candidate_k=20,
                    mmr_lambda=0.7,
                ),
            ],
        )

    def test_interactive_cli_exits_on_eof(self) -> None:
        with (
            patch("builtins.input", side_effect=EOFError),
            patch("builtins.print"),
        ):
            run_interactive()


if __name__ == "__main__":
    unittest.main()
