# react-agent-from-scratch

A minimal [ReAct](https://arxiv.org/abs/2210.03629) (Reason + Act) agent loop in pure Python — no LangChain, no frameworks, standard library only.

The agent reasons step by step, calls tools, observes the results, and repeats until it can answer. The whole pattern fits in ~120 lines of readable code, which makes it a good starting point for understanding how agent frameworks actually work under the hood.

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
- `demo.py` — offline multi-step demo · `run_live.py` — real-model example · `test_agent.py` — tests

## Limitations

- Single string argument per tool call (keeps parsing trivial).
- No memory beyond the current task, no streaming.
- `FakeLLM` is scripted — it's a test double, not intelligence.

## License

MIT
