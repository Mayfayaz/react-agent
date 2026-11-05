"""Demo: the ReAct agent calling tools from an MCP server — no API key needed.

By default it talks to the bundled ``mock_mcp_server.py``. Pass
``--server-cmd`` to point it at any real MCP server instead, e.g. the
``mcp-tool-server`` project (its read-only ``search_notes`` tool)::

    python3 demo_mcp.py --server-cmd "python3 ../mcp-tool-server/server.py"
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from agent import ReActAgent
from mcp_client import MCPClient
from mcp_tools import mcp_tools
from tools import TOOLS

HERE = Path(__file__).parent


class MockLLM:
    """Scripted LLM for the mock server: adds 21 + 21 via the MCP tool."""

    _SCRIPT = [
        "Thought: I need 21 + 21, and the MCP server has an add_numbers tool.\n"
        'Action: add_numbers[{"a": 21, "b": 21}]',
        "Thought: I now know the final answer.\nFinal Answer: 21 + 21 = 42.",
    ]

    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        reply = self._SCRIPT[min(self.calls, len(self._SCRIPT) - 1)]
        self.calls += 1
        return reply


class NotesLLM:
    """Scripted LLM for a notes MCP server: searches, then quotes the result."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return ("Thought: I need to search the notes for MCP.\n"
                    'Action: search_notes[{"query": "mcp"}]')
        observation = next(
            (m["content"] for m in reversed(messages)
             if m["role"] == "user" and m["content"].startswith("Observation:")),
            "Observation: (none)",
        )
        snippet = observation[len("Observation:"):].strip().replace("\n", " ")[:240]
        return (f"Thought: I now know the final answer.\nFinal Answer: "
                f"The notes search returned: {snippet}")


def run_demo(command, cwd, llm, task):
    with MCPClient(command, cwd=cwd) as client:
        tools = {**TOOLS, **mcp_tools(client)}
        print(f"Connected to MCP server: {client.server_info} "
              f"(protocol {client.protocol_version})")
        print(f"Tools available: {', '.join(sorted(tools))}\n")
        agent = ReActAgent(llm, tools)
        result = agent.run(task)
        print(f"Task: {task}\n")
        for i, step in enumerate(result.steps, 1):
            print(f"--- step {i} ---")
            print(step.thought)
            print(f"Observation: {step.observation}\n")
        print(f"Answer: {result.answer}")
        print(f"Converged: {result.converged}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-cmd", default=None,
                        help="command launching an MCP server, e.g. "
                             "'python3 ../mcp-tool-server/server.py'")
    parser.add_argument("--server-cwd", default=None,
                        help="working directory for the server command")
    args = parser.parse_args(argv)

    if args.server_cmd:
        command = shlex.split(args.server_cmd)
        llm, task = NotesLLM(), ("Search my research notes for anything about "
                                 "MCP and summarise what you find.")
    else:
        command = [sys.executable, str(HERE / "mock_mcp_server.py")]
        llm, task = MockLLM(), "What is 21 + 21? Use the MCP server's tools."
    result = run_demo(command, args.server_cwd, llm, task)
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
