import unittest
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from src.chat_rag import (
    MAX_COMPLETION_TOKENS,
    SYSTEM_PROMPT,
    QueryPlan,
    _normalize_chunk,
    _select_retrieval_mode,
    ask_openrouter,
    build_context,
    diversify_candidates,
    is_comparison_query,
    is_high_stakes_query,
    plan_logical_query,
    rerank_candidates,
    reciprocal_rank_fusion,
    retrieve_chunks,
    run_interactive,
    select_best_per_source,
)
from src.spectrag import ChunkGraph


class ChatRagTests(unittest.TestCase):
    def test_auto_retrieval_mode_uses_query_shape_not_stakes(self) -> None:
        def plan(query: str, *, is_complex: bool = False) -> QueryPlan:
            return QueryPlan(
                original_query=query,
                standalone_query=query,
                is_complex=is_complex,
                sub_queries=[query],
            )

        self.assertEqual(
            _select_retrieval_mode(
                "What does the label say about pregnancy?",
                plan("What does the label say about pregnancy?"),
                "hybrid",
                table_only=False,
            ),
            "hybrid",
        )
        self.assertEqual(
            _select_retrieval_mode(
                "Compare revenue across two years",
                plan("Compare revenue across two years"),
                "hybrid",
                table_only=False,
            ),
            "spectrag",
        )
        self.assertEqual(
            _select_retrieval_mode(
                "If the threshold is met, what happens unless an exception applies?",
                plan(
                    "If the threshold is met, what happens unless an exception applies?",
                    is_complex=True,
                ),
                "hybrid",
                table_only=False,
            ),
            "agentic",
        )
        self.assertEqual(
            _select_retrieval_mode(
                "What is shown in Table 2?",
                plan("What is shown in Table 2?"),
                "hybrid",
                table_only=False,
            ),
            "hybrid",
        )
        self.assertEqual(
            _select_retrieval_mode(
                "Compare revenue across two years",
                plan("Compare revenue across two years"),
                "agentic",
                table_only=False,
            ),
            "agentic",
        )

    def test_table_row_context_uses_self_contained_row(self) -> None:
        normalized = _normalize_chunk(
            {
                "id": "row-1",
                "text": "Caption; Columns: Year | Revenue; Row: 2025 | 100",
                "metadata": {
                    "chunk_type": "table_row",
                    "parent_text": "The full table with many rows",
                },
                "rerank_score": 0.9,
            }
        )

        self.assertEqual(normalized["context_text"], normalized["text"])

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
            {"predict": lambda self, pairs, **kwargs: [0.9, 0.1]},
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
            {"predict": lambda self, pairs, **kwargs: [0.2, 0.9]},
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
        self.assertTrue(is_comparison_query("Which of the drug labels says this?"))
        self.assertFalse(is_comparison_query("What is the recommended dose?"))

    def test_heuristic_planner_handles_regulatory_compound_questions(self) -> None:
        claim_check = plan_logical_query(
            "A medium alcohol fuel plant needs a $50,000 bond regardless of size. Is that right?",
            use_llm=False,
        )
        parts_question = plan_logical_query(
            "Which three parts could all apply to a single bottle of vinegar-based sauce, and why?",
            use_llm=False,
        )
        unit_comparison = plan_logical_query(
            "How is 'proof gallon' defined or used in 27 CFR 19, and does 27 CFR 25 use the same unit?",
            use_llm=False,
        )
        aspect_comparison = plan_logical_query(
            "Which parts bear on the color itself and which on the product grade?",
            use_llm=False,
        )
        standalone = plan_logical_query(
            "What does 27 CFR Part 25 say about bottling?",
            use_llm=False,
        )

        self.assertTrue(claim_check.is_complex)
        self.assertEqual(
            claim_check.sub_queries,
            ["A medium alcohol fuel plant needs a $50,000 bond regardless of size"],
        )
        self.assertTrue(parts_question.is_complex)
        self.assertTrue(unit_comparison.is_complex)
        self.assertEqual(len(unit_comparison.sub_queries), 2)
        self.assertIn("27 CFR 19", unit_comparison.sub_queries[0])
        self.assertIn("27 CFR 25", unit_comparison.sub_queries[1])
        self.assertTrue(aspect_comparison.is_complex)
        self.assertEqual(
            aspect_comparison.sub_queries,
            [
                "Which parts bear on the color itself?",
                "Which parts bear on the product grade?",
            ],
        )
        self.assertFalse(standalone.is_complex)

    def test_planner_omits_unrelated_history_for_standalone_question(self) -> None:
        history = [{"query": "Earlier question", "answer": "Earlier answer"}]
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("src.chat_rag._get_openrouter_client") as get_client,
        ):
            plan = plan_logical_query(
                "What are the REMS ANC monitoring requirements for clozapine?",
                history=history,
            )

        get_client.assert_not_called()
        self.assertEqual(len(plan.sub_queries), 1)

    def test_planner_uses_history_for_a_short_follow_up(self) -> None:
        history = [{"query": "Earlier question", "answer": "Earlier answer"}]
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("src.chat_rag._get_openrouter_client", return_value=object()),
            patch(
                "src.chat_rag._json_completion",
                return_value={
                    "standalone_query": "Resolved follow-up question",
                    "is_complex": False,
                    "sub_queries": ["Unneeded alternate query"],
                    "logical_conditions": ["Unneeded condition"],
                },
            ) as json_completion,
        ):
            plan = plan_logical_query("What about it?", history=history)

        prompt = json_completion.call_args.kwargs["user_prompt"]
        self.assertIn("PRIOR CONVERSATION", prompt)
        self.assertEqual(plan.standalone_query, "Resolved follow-up question")
        self.assertEqual(plan.sub_queries, ["Resolved follow-up question"])
        self.assertEqual(plan.logical_conditions, [])

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

    def test_spectrag_retrieval_expands_and_returns_graph_scores(self) -> None:
        metadata = {
            "source": "report.pdf",
            "domain": "financial",
            "pdf_page": 1,
            "citation": "[report.pdf p.1]",
            "chunk_type": "prose",
            "section_path": ["Results"],
        }
        dense_hits = [
            {"id": "seed", "text": "revenue increased", "metadata": metadata}
        ]
        candidates = [
            {"id": "seed", "text": "revenue increased", "metadata": metadata},
            {
                "id": "neighbor",
                "text": "costs increased",
                "metadata": metadata,
            },
        ]
        graph = ChunkGraph(
            ids=["seed", "neighbor"],
            adjacency=__import__("scipy").sparse.csr_matrix(
                [[0.0, 1.0], [1.0, 0.0]]
            ),
            coordinates=__import__("numpy").zeros((2, 1)),
        )

        def fake_rerank(query, hits, **kwargs):
            return [
                {**hit, "rerank_score": 0.9 - index * 0.1}
                for index, hit in enumerate(hits)
            ]

        with (
            patch("src.chat_rag.query_index", return_value=dense_hits),
            patch("src.chat_rag.query_lexical_index", return_value=[]),
            patch("src.chat_rag.ChunkGraph.load", return_value=graph),
            patch("src.chat_rag.get_index_chunks", return_value=candidates),
            patch("src.chat_rag.rerank_candidates", side_effect=fake_rerank),
        ):
            chunks = retrieve_chunks(
                "revenue",
                2,
                index_dir=Path("test-index"),
                candidate_k=2,
                retrieval_mode="spectrag",
            )

        self.assertEqual({chunk["id"] for chunk in chunks}, {"seed", "neighbor"})
        self.assertTrue(all("spectrag_score" in chunk for chunk in chunks))
        self.assertTrue(all("ppr_score" in chunk for chunk in chunks))

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

        self.assertEqual(
            create.call_args.kwargs["max_tokens"], MAX_COMPLETION_TOKENS
        )
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

    def test_openrouter_recovers_a_truncated_answer(self) -> None:
        responses = [
            SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Partial"),
                        finish_reason="length",
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Fact [a.pdf p.1]"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
        create = Mock(side_effect=responses)
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
            patch.dict(
                "os.environ",
                {"OPENROUTER_API_KEY": "test-key-truncated-recovery"},
            ),
        ):
            result = ask_openrouter("question", chunks)

        self.assertEqual(create.call_count, 2)
        self.assertEqual(result["answer"], "Fact [a.pdf p.1]")
        self.assertTrue(result["generation_complete"])
        self.assertEqual(result["finish_reason"], "stop")

    def test_openrouter_does_not_return_an_unrecovered_partial_answer(self) -> None:
        responses = [
            SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Partial answer"),
                        finish_reason="length",
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Still partial"),
                        finish_reason="length",
                    )
                ],
                usage=None,
            ),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(side_effect=responses))
            )
        )
        openai_module = ModuleType("openai")
        openai_module.OpenAI = Mock(return_value=client)

        with (
            patch.dict("sys.modules", {"openai": openai_module}),
            patch.dict(
                "os.environ",
                {"OPENROUTER_API_KEY": "test-key-truncated-fail"},
            ),
        ):
            result = ask_openrouter("question", [])

        self.assertFalse(result["generation_complete"])
        self.assertNotIn("Partial answer", result["answer"])
        self.assertIn("couldn't produce a complete answer", result["answer"])

    def test_cfr_questions_require_claim_support_verification(self) -> None:
        self.assertTrue(is_high_stakes_query("What does 21 CFR Part 73 require?"))

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
            patch(
                "builtins.input",
                side_effect=[
                    "write a briefing",
                    "second question",
                    "quit",
                ],
            ),
            patch(
                "src.chat_rag.route_mcp_tools",
                side_effect=["Generated document: briefing.pdf", None],
            ) as route,
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

        self.assertEqual(route.call_count, 2)
        self.assertEqual(route.call_args_list[0].args, ("write a briefing",))
        self.assertEqual(route.call_args_list[1].args, ("second question",))
        self.assertEqual(run.call_count, 1)
        rag_call = run.call_args_list[0]
        self.assertEqual(rag_call.args, ("second question", False))
        self.assertEqual(rag_call.kwargs["retrieval_mode"], "hybrid")

    def test_interactive_cli_returns_mcp_response_without_rag(self) -> None:
        with (
            patch(
                "builtins.input",
                side_effect=["write a briefing", "quit"],
            ),
            patch(
                "src.chat_rag.route_mcp_tools",
                return_value="Generated document: briefing.pdf",
            ) as route,
            patch("src.chat_rag.run_one") as run,
            patch("builtins.print"),
        ):
            run_interactive(index_dir=Path("test-index"))

        route.assert_called_once()
        run.assert_not_called()

    def test_one_shot_cli_routes_mcp_tools_before_rag(self) -> None:
        with (
            patch("sys.argv", ["chat_rag.py", "write a briefing"]),
            patch(
                "src.chat_rag.route_mcp_tools",
                return_value="Generated document: briefing.pdf",
            ) as route,
            patch("src.chat_rag.run_one") as run,
            patch("builtins.print"),
        ):
            from src.chat_rag import main

            main()

        route.assert_called_once()
        run.assert_not_called()

    def test_interactive_cli_exits_on_eof(self) -> None:
        with (
            patch("builtins.input", side_effect=EOFError),
            patch("builtins.print"),
        ):
            run_interactive()


if __name__ == "__main__":
    unittest.main()
