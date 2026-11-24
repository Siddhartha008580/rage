import unittest
from pathlib import Path
from unittest.mock import patch

from src.mcp_tools import has_tool_intent, route_mcp_tools


class McpToolsTests(unittest.TestCase):
    def test_common_document_creation_requests_trigger_mcp_routing(self) -> None:
        requests = (
            'Create a nutrient-claim evaluation worksheet for "reduced sodium" '
            'and "lean" claims on a ground beef product.',
            "Put together a compliance checklist for this policy.",
            "Turn these notes into a PDF guide.",
            "Save this as a PDF report.",
        )

        for request in requests:
            with self.subTest(request=request):
                self.assertTrue(has_tool_intent(request))

    def test_factual_document_questions_do_not_trigger_generation(self) -> None:
        queries = (
            "Which documents mention worksheet requirements?",
            "When must a party prepare a report?",
            "How do I create a compliance worksheet?",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertFalse(has_tool_intent(query))

    def test_route_reports_cause_from_single_exception_group(self) -> None:
        async def fail(_query: str, **_kwargs: object) -> str | None:
            raise ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [RuntimeError("429 Too Many Requests")],
            )

        with (
            patch("src.mcp_tools.has_tool_intent", return_value=True),
            patch("src.mcp_tools._route_mcp_tools", fail),
            self.assertRaisesRegex(
                RuntimeError,
                r"MCP tool routing failed \(RuntimeError\): 429 Too Many Requests",
            ),
        ):
            route_mcp_tools(
                "write a brief",
                index_dir=Path("index"),
                model="test-model",
                reasoning_effort="none",
            )


if __name__ == "__main__":
    unittest.main()
