import unittest

from src.ingest import _bbox_iou, _merge_table_transformer_predictions
from src.table_transformer import _normalize_processor_size, _table_html


class TableTransformerTests(unittest.TestCase):
    def test_legacy_long_edge_processor_size_is_normalized(self) -> None:
        processor = type("Processor", (), {"size": {"longest_edge": 800}})()

        _normalize_processor_size(processor)

        self.assertEqual(
            processor.size,
            {"max_height": 800, "max_width": 800},
        )

    def test_size_mapping_object_is_normalized(self) -> None:
        class SizeMapping:
            def __init__(self) -> None:
                self.values = {"longest_edge": 800, "shortest_edge": None}

            def get(self, key: str) -> int | None:
                return self.values.get(key)

        processor = type("Processor", (), {"size": SizeMapping()})()

        _normalize_processor_size(processor)

        self.assertEqual(
            processor.size,
            {"max_height": 800, "max_width": 800},
        )

    def test_bbox_iou_is_geometric(self) -> None:
        self.assertAlmostEqual(_bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(_bbox_iou((0, 0, 10, 10), (11, 0, 20, 10)), 0)

    def test_hybrid_replaces_overlapping_table_only_at_confidence(self) -> None:
        fallback = {
            0: [{"bbox": (0, 0, 100, 100), "rows": [["old"]], "table_index": 1}]
        }
        high_confidence = {
            0: [
                {
                    "bbox": (5, 5, 95, 95),
                    "rows": [["new"]],
                    "html": "<table></table>",
                    "confidence": 0.8,
                    "detection_score": 0.9,
                    "structure_score": 0.8,
                    "word_coverage": 0.7,
                }
            ]
        }

        merged = _merge_table_transformer_predictions(fallback, high_confidence)

        self.assertEqual(merged[0][0]["rows"], [["new"]])
        self.assertEqual(merged[0][0]["extractor"], "table-transformer")

        low_confidence = {0: [{**high_confidence[0][0], "confidence": 0.2}]}
        preserved = _merge_table_transformer_predictions(fallback, low_confidence)
        self.assertEqual(preserved[0][0]["rows"], [["old"]])

    def test_unmatched_high_confidence_table_is_added(self) -> None:
        fallback = {0: [{"bbox": (0, 0, 10, 10), "rows": [["existing"]]}]}
        transformed = {
            0: [
                {
                    "bbox": (30, 30, 90, 90),
                    "rows": [["detected"]],
                    "confidence": 0.9,
                }
            ]
        }

        merged = _merge_table_transformer_predictions(fallback, transformed)

        self.assertEqual(len(merged[0]), 2)
        self.assertEqual(merged[0][1]["rows"], [["detected"]])

    def test_html_serialization_preserves_spans_and_escapes_text(self) -> None:
        rendered = _table_html(
            [["North & West", ""], ["A < B", "C"]],
            {(0, 0): (1, 2)},
            {0},
        )

        self.assertIn('<th colspan="2">North &amp; West</th>', rendered)
        self.assertIn("A &lt; B", rendered)
        self.assertNotIn("<th></th>", rendered)


if __name__ == "__main__":
    unittest.main()