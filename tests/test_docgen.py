import unittest

from src.docgen import _validate_outline, assemble


class DocgenTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
