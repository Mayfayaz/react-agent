"""A tiny MCP server over stdio — standard library only.

Test fixture and offline demo target for ``mcp_client.py``: it speaks the
same JSON-RPC 2.0 / stdio transport as a real MCP server, so the client and
the ReAct agent can be exercised with no network and no extra packages.

Tools:
- ``add_numbers`` — add two numbers: ``{"a": 21, "b": 21}``
- ``reverse_text`` — reverse a string: ``{"text": "hello"}``

Run it directly to serve on stdin/stdout::

    python3 mock_mcp_server.py
"""

from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2024-11-05"


def _tool_defs():
    return [
        {
            "name": "add_numbers",
            "description": "Add two numbers, e.g. add_numbers[{\"a\": 21, \"b\": 21}].",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "reverse_text",
            "description": "Reverse a string, e.g. reverse_text[hello].",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    ]


def _text_result(text):
    return {"content": [{"type": "text", "text": str(text)}]}


def _call_tool(name, arguments):
    if name == "add_numbers":
        return _text_result(arguments["a"] + arguments["b"])
    if name == "reverse_text":
        return _text_result(arguments["text"][::-1])
    return {"content": [{"type": "text", "text": f"unknown tool: {name}"}],
            "isError": True}


def _handle(message):
    """Return the JSON-RPC response dict, or None for notifications."""
    method = message.get("method")
    msg_id = message.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "serverInfo": {"name": "mock-mcp-server", "version": "0.1.0"},
        }}
    if method == "notifications/initialized":
        return None  # notification: no reply
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": _tool_defs()}}
    if method == "tools/call":
        params = message.get("params", {})
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": _call_tool(params.get("name"), params.get("arguments", {}))}
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {
            "code": -32601, "message": f"method not found: {method}"}}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
