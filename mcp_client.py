"""A minimal MCP (Model Context Protocol) client over stdio — standard library only.

Speaks JSON-RPC 2.0 over a server subprocess's stdin/stdout, the same
transport the official MCP Python SDK uses for local servers. No extra
dependencies: this lets the ReAct agent call tools from *any* MCP server —
including the ``mcp-tool-server`` project — without pulling in the SDK.

Protocol flow (MCP 2024-11-05):
    1. ``initialize`` request (protocol version, capabilities, client info)
    2. ``notifications/initialized`` notification
    3. ``tools/list`` and ``tools/call`` requests as needed

Usage::

    from mcp_client import MCPClient

    with MCPClient(["python3", "server.py"], cwd="/path/to/server") as client:
        print(client.list_tools())              # -> list[MCPTool]
        print(client.call_tool("search_notes", {"query": "mcp"}))
"""

from __future__ import annotations

import json
import select
import subprocess
import sys

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "react-agent-mcp-client"
CLIENT_VERSION = "0.1.0"


class MCPError(Exception):
    """The server returned a JSON-RPC error, or the protocol exchange failed."""


class MCPTool:
    """One tool advertised by an MCP server (name, description, JSON schema)."""

    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description or ""
        self.input_schema = input_schema or {}

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
        )

    @property
    def required_params(self):
        """Names of the tool's required arguments, in schema order."""
        return list(self.input_schema.get("required", []))

    @property
    def string_params(self):
        """Names of the tool's string-typed arguments, in schema order."""
        props = self.input_schema.get("properties", {})
        return [k for k, v in props.items() if v.get("type") == "string"]

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"MCPTool({self.name!r})"


class MCPClient:
    """JSON-RPC client for one MCP server process over stdio.

    Use as a context manager so the server subprocess is always stopped::

        with MCPClient(["python3", "server.py"]) as client:
            ...
    """

    def __init__(self, command, cwd=None, env=None, timeout=30.0):
        self.command = list(command)
        self.cwd = cwd
        self.env = env
        self.timeout = timeout
        self._proc = None
        self._next_id = 0
        self.server_info = {}
        self.protocol_version = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self):
        """Start the server and run the MCP initialize handshake."""
        if self._proc is not None:
            raise MCPError("already connected")
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # server logs must not block the pipe
            text=True,
            bufsize=1,
            cwd=self.cwd,
            env=self.env,
        )
        try:
            result = self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            })
        except Exception:
            self.close()
            raise
        self.protocol_version = result.get("protocolVersion")
        self.server_info = result.get("serverInfo", {})
        self._notify("notifications/initialized")
        return self

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Terminate the server process (no-op if already closed)."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

    # -- tools -------------------------------------------------------------
    def list_tools(self):
        """Return the server's tools as a list of :class:`MCPTool`."""
        result = self._request("tools/list", {})
        return [MCPTool.from_dict(t) for t in result.get("tools", [])]

    def call_tool(self, name, arguments=None):
        """Call a tool and return its text content as a single string.

        Raises :class:`MCPError` if the server reports an error or the tool
        itself returns ``isError: true``.
        """
        result = self._request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        if result.get("isError"):
            text = _content_text(result.get("content", []))
            raise MCPError(f"tool {name!r} failed: {text}")
        return _content_text(result.get("content", []))

    # -- JSON-RPC plumbing ---------------------------------------------------
    def _request(self, method, params):
        self._next_id += 1
        request_id = self._next_id
        self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })
        return self._read_response(request_id)

    def _notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def _send(self, message):
        if self._proc is None or self._proc.stdin is None:
            raise MCPError("not connected")
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise MCPError("server closed stdin (did it exit?)") from e

    def _read_response(self, request_id):
        """Read lines until the response matching ``request_id`` arrives.

        Notifications and responses to other requests are skipped, so a
        chatty server cannot confuse the exchange.
        """
        if self._proc is None or self._proc.stdout is None:
            raise MCPError("not connected")
        fd = self._proc.stdout.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], self.timeout)
            if not ready:
                raise MCPError(f"timed out after {self.timeout}s waiting for {request_id}")
            line = self._proc.stdout.readline()
            if line == "":  # EOF: the server exited mid-conversation
                raise MCPError("server exited while waiting for a response")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore non-JSON lines (e.g. stray server output)
            if message.get("id") != request_id:
                continue  # notification, or a response to another request
            if "error" in message:
                err = message["error"]
                raise MCPError(f"{err.get('message', err)} (code {err.get('code')})")
            return message.get("result", {})


def _content_text(content):
    """Join the text blocks of a tools/call result into one string."""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, dict):
            parts.append(json.dumps(block))
        else:
            parts.append(str(block))
    return "\n".join(parts)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    with MCPClient(sys.argv[1:]) as client:
        print(f"server: {client.server_info} (protocol {client.protocol_version})")
        for tool in client.list_tools():
            print(f"  - {tool.name}: {tool.description}")
