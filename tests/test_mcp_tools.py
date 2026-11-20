import unittest
from pathlib import Path
from unittest.mock import patch

from src.mcp_tools import route_mcp_tools


class McpToolsTests(unittest.TestCase):
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
