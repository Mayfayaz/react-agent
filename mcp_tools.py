"""Adapt an MCP server's tools to the ReAct agent's ``dict[str, callable]``.

The agent's tools each take a single string argument. MCP tools take a
JSON object of named arguments, so the adapter bridges the two:

- If the argument parses as a JSON object, it is passed through as-is.
- Otherwise, if the tool has exactly one string parameter, the raw argument
  becomes that parameter's value — so ``Action: reverse_text[hello]`` works.
- Otherwise the call fails with a clear error the agent can see and recover
  from (the error is returned as an Observation, not raised).
"""

from __future__ import annotations

import json

from mcp_client import MCPClient, MCPError


def _coerce_arguments(tool, argument):
    """Turn one string argument into the dict an MCP tool expects."""
    argument = (argument or "").strip()
    if not argument:
        return {}
    try:
        parsed = json.loads(argument)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if len(tool.string_params) == 1:
        return {tool.string_params[0]: argument}
    raise MCPError(
        f"tool {tool.name!r} needs a JSON object argument, e.g. "
        f"{tool.name}[{{\"...\": ...}}]; got {argument!r}"
    )


def mcp_tools(client):
    """Build ``{name: callable}`` for every tool the connected client offers.

    Each callable takes a single string argument (see module docstring) and
    returns the tool's text result. Tool descriptions from the server become
    the callables' docstrings, so the agent sees them in its system prompt.
    """
    tools = {}
    for tool in client.list_tools():
        tools[tool.name] = _make_tool_fn(client, tool)
    return tools


def _make_tool_fn(client, tool):
    def _tool(argument: str) -> str:
        try:
            arguments = _coerce_arguments(tool, argument)
            return client.call_tool(tool.name, arguments)
        except (MCPError, ValueError) as e:
            return f"Error: {e}"

    _tool.__doc__ = tool.description or f"MCP tool {tool.name}."
    _tool.__name__ = tool.name
    return _tool


def connect_mcp_tools(command, cwd=None, env=None):
    """Connect to an MCP server and return ``(client, tools)``.

    The caller owns the client and must close it (or use it as a context
    manager) when the agent is done::

        client, tools = connect_mcp_tools(["python3", "server.py"])
        agent = ReActAgent(llm, {**TOOLS, **tools})
        ...
        client.close()
    """
    client = MCPClient(command, cwd=cwd, env=env).connect()
    return client, mcp_tools(client)
