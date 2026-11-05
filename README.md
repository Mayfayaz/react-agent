# react-agent-from-scratch

A minimal [ReAct](https://arxiv.org/abs/2210.03629) (Reason + Act) agent loop in pure Python — no LangChain, no frameworks, standard library only.

The agent reasons step by step, calls tools, observes the results, and repeats until it can answer. The whole pattern fits in ~120 lines of readable code, which makes it a good starting point for understanding how agent frameworks actually work under the hood.

It can also call tools from any [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server: `mcp_client.py` is a stdlib-only MCP client speaking JSON-RPC over stdio, so the agent gains every tool a server offers with no extra dependency.

## How it works

```
User task
   │
   ▼
┌─────────────┐   Thought: ...     ┌──────┐
│  ReAct loop │ ──────────────────▶│ LLM  │
│  (agent.py) │ ◀──────────────────│      │
└─────────────┘  Action: tool[arg] └──────┘
   │ ▲
   │ │ Observation: <tool result>
   ▼ │
┌─────────────┐
│ tools.py    │  calculator · unit_convert · get_time · word_count
└─────────────┘
```

Each iteration the LLM emits either `Action: <tool>[<argument>]` or `Final Answer: <...>`. The runner executes the tool, feeds back the `Observation`, and loops (max 8 steps). Off-format replies get a corrective nudge instead of crashing the run.

## Quickstart — no API key needed

```bash
python3 demo.py                 # offline demo with a scripted LLM
python3 -m unittest test_agent -v
```

## With a real model

```bash
export OPENAI_API_KEY=sk-...
python3 run_live.py             # any OpenAI-compatible endpoint; override with MODEL= / OPENAI_BASE_URL=
```

## Calling tools from an MCP server

`mcp_client.py` speaks the MCP initialize / `tools/list` / `tools/call`
exchange over a server subprocess's stdio — the same transport the official
SDK uses — using only the standard library. `mcp_tools.py` adapts the
server's tools to the agent's `dict[str, callable]` interface, so local and
remote tools mix freely and server tool descriptions land in the system
prompt automatically.

```bash
python3 demo_mcp.py                 # offline demo against the bundled mock server
python3 -m unittest test_mcp_client -v
```

Point it at a real server — for example the `mcp-tool-server` project, whose
`search_notes` / `read_note` / `add_note` tools become agent tools:

```bash
python3 demo_mcp.py --server-cmd "python3 ../mcp-tool-server/server.py"
```

Or wire it up in code:

```python
from agent import ReActAgent
from mcp_tools import connect_mcp_tools
from tools import TOOLS

client, mcp = connect_mcp_tools(["python3", "server.py"])
agent = ReActAgent(llm, {**TOOLS, **mcp})
result = agent.run("Search my notes for MCP and summarise what you find.")
client.close()
```

MCP tools take a JSON object of named arguments, so the agent writes
`Action: search_notes[{"query": "mcp"}]`. As a shorthand, a tool with
exactly one string parameter also accepts the raw string:
`Action: reverse_text[hello]`.

## Add your own tool

```python
# tools.py
def roll_dice(argument: str) -> str:
    """Roll dice, e.g. roll_dice[2d6]."""
    ...

TOOLS["roll_dice"] = roll_dice
```

That's it — the agent picks it up automatically because tool docstrings are injected into the system prompt. Each tool takes a single string argument by design, which keeps action parsing trivial.

## Project layout

- `agent.py` — the ReAct loop, action parsing, step trace
- `llm.py` — OpenAI-compatible client (urllib) + `FakeLLM`, a scripted stand-in for offline runs
- `tools.py` — tools: safe AST-based calculator (no `eval`), unit converter, clock, word counter
- `mcp_client.py` — stdlib-only MCP client over stdio (initialize / tools/list / tools/call)
- `mcp_tools.py` — adapts an MCP server's tools to the agent's tool interface
- `mock_mcp_server.py` — tiny stdio MCP server used by the demo and tests
- `demo.py` — offline multi-step demo · `demo_mcp.py` — MCP demo (mock server by default, `--server-cmd` for a real one) · `run_live.py` — real-model example · `test_agent.py` — tests · `test_mcp_client.py` — MCP client/adapter tests

## Limitations

- Single string argument per tool call (keeps parsing trivial). MCP tools take a JSON object; a single-string-parameter tool also accepts the raw string.
- MCP support covers the stdio transport only (the standard transport for local servers); no HTTP/SSE transport yet.
- No memory beyond the current task, no streaming.
- `FakeLLM` is scripted — it's a test double, not intelligence.

## License

MIT
