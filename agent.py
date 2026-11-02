"""A minimal ReAct (Reason + Act) agent loop. No frameworks, stdlib only."""

import re
from dataclasses import dataclass, field

_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]\s*$", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)

SYSTEM_PROMPT = """You solve tasks by reasoning step by step and calling tools.

In each step, output exactly one of:

Thought: <your reasoning about what to do next>
Action: <tool_name>[<argument>]

After each Action, you will receive:
Observation: <tool result>

When you have everything needed, output:
Thought: I now know the final answer.
Final Answer: <your answer>

Rules:
- Never invent an Observation; only the runner provides them.
- One Action per step. Keep the argument on a single line.
- Available tools:
{tool_docs}
"""


@dataclass
class Step:
    thought: str = ""
    action: str = ""
    argument: str = ""
    observation: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)
    converged: bool = False


class ReActAgent:
    """Runs the Thought -> Action -> Observation loop until a Final Answer."""

    def __init__(self, llm, tools, max_steps=8):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def _tool_docs(self):
        return "\n".join(f"- {name}: {fn.__doc__ or ''}".strip() for name, fn in self.tools.items())

    def run(self, task):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tool_docs=self._tool_docs())},
            {"role": "user", "content": task},
        ]
        steps = []
        for _ in range(self.max_steps):
            reply = self.llm.chat(messages).strip()
            messages.append({"role": "assistant", "content": reply})

            final = _FINAL_RE.search(reply)
            if final:
                return AgentResult(answer=final.group(1).strip(), steps=steps, converged=True)

            m = _ACTION_RE.search(reply)
            if not m:
                messages.append({
                    "role": "user",
                    "content": ("Observation: Format error — reply with 'Thought:' followed by "
                                "'Action: name[argument]', or 'Final Answer:'."),
                })
                continue

            name, arg = m.group(1), m.group(2).strip()
            step = Step(thought=reply, action=name, argument=arg)
            tool = self.tools.get(name)
            if tool is None:
                step.observation = f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}"
            else:
                try:
                    step.observation = str(tool(arg))
                except Exception as e:  # surface tool errors back to the agent
                    step.observation = f"Error: {e}"
            steps.append(step)
            messages.append({"role": "user", "content": f"Observation: {step.observation}"})

        return AgentResult(answer="Max steps reached without a final answer.", steps=steps, converged=False)
