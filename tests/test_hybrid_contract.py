from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from src.contracts import PageArtifact
from src.quality.vlm_gate import vlm_divergence
from src.routing import load_routing_config
from src.tables.rich_html import parse_html_table_rich


class HybridContractTests(unittest.TestCase):
    def test_page_artifact_is_frozen(self) -> None:
        artifact = PageArtifact(
            page_number=1,
            route="native",
            markdown="text",
            tables=[],
            figures=[],
            native_links=[],
            confidence=1.0,
            reason_codes=[],
        )
        with self.assertRaises(FrozenInstanceError):
            artifact.markdown = "changed"

    def test_rich_span_grid(self) -> None:
        grid = parse_html_table_rich(
            '<table><tr><th rowspan="2">A</th><th colspan="3">B</th></tr>'
            '<tr><td>C</td></tr></table>'
        )
        self.assertIsNotNone(grid[0][0])
        self.assertIsNone(grid[1][0])
        self.assertIsNotNone(grid[0][1])
        self.assertIsNone(grid[0][2])
        self.assertIsNone(grid[0][3])

    def test_vlm_divergence_rejects_hallucination(self) -> None:
        ratio, codes = vlm_divergence(
            "Revenue grew 12% in Q3. The company also opened 47 new offices in Antarctica.",
            "Revenue grew 12% in Q3.",
        )
        self.assertGreater(ratio, 0.15)
        self.assertIn("VLM_DIVERGENCE", codes)

    def test_routing_config(self) -> None:
        config = load_routing_config(Path("config/routing.yaml"))
        self.assertEqual(config["docling_confidence_threshold"], 0.55)


if __name__ == "__main__":
    unittest.main()
