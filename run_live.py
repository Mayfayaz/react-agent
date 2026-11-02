"""Live run against any OpenAI-compatible API. Needs OPENAI_API_KEY.

Optional env vars: MODEL (default gpt-4o-mini), OPENAI_BASE_URL.
"""

import os

from agent import ReActAgent
from llm import OpenAICompatibleLLM
from tools import TOOLS


def main():
    llm = OpenAICompatibleLLM(model=os.environ.get("MODEL", "gpt-4o-mini"))
    agent = ReActAgent(llm, TOOLS)
    task = "A store sells 3 apples for $2.00. How much do 17 apples cost? Round to cents."
    result = agent.run(task)
    print(f"Task: {task}\n")
    for i, step in enumerate(result.steps, 1):
        print(f"--- step {i} ---")
        print(step.thought)
        print(f"Observation: {step.observation}\n")
    print(f"Answer: {result.answer}")


if __name__ == "__main__":
    main()
