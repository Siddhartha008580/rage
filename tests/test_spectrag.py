import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.spectrag import (
    ChunkGraph,
    build_chunk_graph,
    count_tokens,
    expand_candidate_ids,
    graph_relevance_scores,
    generate_utility_questions,
    load_utility_questions,
    select_under_token_budget,
    utility_question_hits,
)


class SpectragGraphTests(unittest.TestCase):
    def test_graph_combines_semantic_section_and_adjacency_signals(self) -> None:
        chunks = [
            {
                "chunk_id": "a",
                "text": "Revenue grew in 2024.",
                "source": "report.pdf",
                "section_path": ["Results", "Revenue"],
                "char_start": 0,
            },
            {
                "chunk_id": "b",
                "text": "Costs also grew in 2024.",
                "source": "report.pdf",
                "section_path": ["Results", "Revenue"],
                "char_start": 100,
            },
            {
                "chunk_id": "c",
                "text": "Unrelated clinical information.",
                "source": "medical.pdf",
                "section_path": ["Safety"],
                "char_start": 0,
            },
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
        )

        graph = build_chunk_graph(chunks, embeddings, dimensions=2)

        self.assertEqual(graph.ids, ["a", "b", "c"])
        self.assertGreater(graph.adjacency[0, 1], 0)
        self.assertEqual(graph.adjacency[0, 2], 0)
        self.assertEqual(graph.coordinates.shape, (3, 2))

    def test_graph_round_trips_without_pickle(self) -> None:
        graph = build_chunk_graph(
            [
                {"chunk_id": "a", "text": "Alpha", "source": "a.pdf"},
                {"chunk_id": "b", "text": "Beta", "source": "b.pdf"},
            ],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            dimensions=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            graph.save(Path(directory))
            loaded = ChunkGraph.load(Path(directory))

        self.assertEqual(loaded.ids, graph.ids)
        np.testing.assert_allclose(loaded.adjacency.toarray(), graph.adjacency.toarray())
        np.testing.assert_allclose(loaded.coordinates, graph.coordinates)

    def test_utility_questions_are_persisted_and_queryable(self) -> None:
        graph = build_chunk_graph(
            [
                {"chunk_id": "a", "text": "risk", "source": "a.pdf"},
                {"chunk_id": "b", "text": "sales", "source": "b.pdf"},
            ],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            utility_embeddings=np.asarray(
                [[0.8, 0.2], [0.1, 0.9]], dtype=np.float32
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            graph.save(Path(directory))
            loaded = ChunkGraph.load(Path(directory))

        hits = utility_question_hits(loaded, np.asarray([1.0, 0.0]), 1)
        self.assertEqual(hits[0]["id"], "a")

    def test_utility_question_generation_is_resumable(self) -> None:
        completion = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"items":[{"chunk_id":"a","questions":'
                                '["What changed?","Why did it change?"]}]}'
                            )
                        )
                    )
                ]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=completion)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            chunks = [{"chunk_id": "a", "text": "A source passage."}]

            generated = generate_utility_questions(chunks, path, client=client)
            repeated = generate_utility_questions(chunks, path, client=client)
            questions = load_utility_questions(path)

        self.assertEqual(generated, 1)
        self.assertEqual(repeated, 0)
        self.assertEqual(len(completion.call_args_list), 1)
        self.assertEqual(len(questions["a"]), 2)

    def test_expansion_and_diffusion_reach_connected_chunks(self) -> None:
        chunks = [
            {"chunk_id": str(index), "text": f"Passage {index}", "source": "doc.pdf"}
            for index in range(4)
        ]
        graph = build_chunk_graph(
            chunks,
            np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
                dtype=np.float32,
            ),
            semantic_neighbors=1,
            dimensions=2,
        )

        expanded = expand_candidate_ids(graph, ["0"], max_hops=2)
        scores = graph_relevance_scores(graph, expanded, {"0": 1.0})

        self.assertIn("1", expanded)
        self.assertGreater(scores["1"]["ppr_score"], 0)
        self.assertGreaterEqual(scores["1"]["spectral_score"], 0)

    def test_selector_respects_rendered_token_budget(self) -> None:
        graph = ChunkGraph(
            ids=["a", "b", "c"],
            adjacency=build_chunk_graph(
                [
                    {"chunk_id": key, "text": key, "source": key}
                    for key in ("a", "b", "c")
                ],
                np.eye(3, dtype=np.float32),
            ).adjacency,
            coordinates=np.zeros((3, 2), dtype=np.float32),
        )
        candidates = [
            {"id": key, "text": "one two three", "source": key,
             "spectrag_score": score, "section_path": [key]}
            for key, score in (("a", 0.9), ("b", 0.8), ("c", 0.7))
        ]
        renderer = lambda selected: " ".join(item["text"] for item in selected)

        selected = select_under_token_budget(
            candidates,
            graph,
            7,
            context_renderer=renderer,
            max_chunks=3,
            token_counter=lambda text: len(text.split()),
        )

        self.assertLessEqual(
            len(renderer(selected).split()), 7
        )
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()