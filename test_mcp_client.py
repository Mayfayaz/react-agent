"""Tests for the MCP client, the ReAct adapter, and the mock server.

All offline, no API key: the mock server is a real subprocess speaking
JSON-RPC over stdio, so these tests exercise the actual protocol exchange.
"""

import sys
import unittest
from pathlib import Path

from agent import ReActAgent
from mcp_client import MCPClient, MCPError
from mcp_tools import _coerce_arguments, mcp_tools

HERE = Path(__file__).parent
MOCK_SERVER = [sys.executable, str(HERE / "mock_mcp_server.py")]


class MockLLM:
    """Scripted LLM: adds 21 + 21 through the MCP tool, then answers."""

    _SCRIPT = [
        "Thought: use the MCP tool.\nAction: add_numbers[{\"a\": 21, \"b\": 21}]",
        "Thought: I now know the final answer.\nFinal Answer: 21 + 21 = 42.",
    ]

    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        reply = self._SCRIPT[min(self.calls, len(self._SCRIPT) - 1)]
        self.calls += 1
        return reply


class TestMCPClient(unittest.TestCase):
    def test_lists_tools(self):
        with MCPClient(MOCK_SERVER) as client:
            tools = client.list_tools()
            self.assertEqual([t.name for t in tools], ["add_numbers", "reverse_text"])
            self.assertEqual(client.server_info["name"], "mock-mcp-server")
            self.assertEqual(client.protocol_version, "2024-11-05")
            add = tools[0]
            self.assertEqual(add.required_params, ["a", "b"])
            self.assertEqual(add.string_params, [])
            self.assertEqual(tools[1].string_params, ["text"])

    def test_call_tool(self):
        with MCPClient(MOCK_SERVER) as client:
            self.assertEqual(client.call_tool("add_numbers", {"a": 21, "b": 21}), "42")
            self.assertEqual(client.call_tool("reverse_text", {"text": "hello"}), "olleh")

    def test_unknown_tool_raises(self):
        with MCPClient(MOCK_SERVER) as client:
            with self.assertRaises(MCPError):
                client.call_tool("nope", {})

    def test_unknown_method_raises(self):
        with MCPClient(MOCK_SERVER) as client:
            with self.assertRaises(MCPError):
                client._request("bogus/method", {})

    def test_server_exiting_early_raises(self):
        with self.assertRaises(MCPError):
            MCPClient([sys.executable, "-c", "import sys; sys.exit(1)"]).connect()

    def test_context_manager_closes_process(self):
        client = MCPClient(MOCK_SERVER)
        with client:
            proc = client._proc
            self.assertIsNone(proc.poll())
        self.assertIsNotNone(proc.poll())  # terminated on exit


class TestMCPToolsAdapter(unittest.TestCase):
    def test_json_object_argument_passed_through(self):
        with MCPClient(MOCK_SERVER) as client:
            tools = mcp_tools(client)
            self.assertEqual(tools["add_numbers"]('{"a": 3, "b": 4}'), "7")

    def test_plain_string_wraps_single_string_param(self):
        with MCPClient(MOCK_SERVER) as client:
            tools = mcp_tools(client)
            self.assertEqual(tools["reverse_text"]("hello"), "olleh")

    def test_bad_argument_returns_error_string(self):
        with MCPClient(MOCK_SERVER) as client:
            tools = mcp_tools(client)
            result = tools["add_numbers"]("not json and not a single param")
            self.assertTrue(result.startswith("Error:"))
            self.assertIn("JSON object", result)

    def test_coerce_empty_argument(self):
        with MCPClient(MOCK_SERVER) as client:
            tool = client.list_tools()[0]
            self.assertEqual(_coerce_arguments(tool, ""), {})

    def test_tool_docstring_comes_from_server(self):
        with MCPClient(MOCK_SERVER) as client:
            tools = mcp_tools(client)
            self.assertIn("Add two numbers", tools["add_numbers"].__doc__)


class TestAgentWithMCPTools(unittest.TestCase):
    def test_agent_calls_mcp_tool_end_to_end(self):
        from tools import TOOLS

        with MCPClient(MOCK_SERVER) as client:
            agent = ReActAgent(MockLLM(), {**TOOLS, **mcp_tools(client)})
            result = agent.run("What is 21 + 21? Use the MCP server's tools.")
            self.assertTrue(result.converged)
            self.assertIn("42", result.answer)
            self.assertEqual(result.steps[0].action, "add_numbers")
            self.assertEqual(result.steps[0].observation, "42")


if __name__ == "__main__":
    unittest.main()
