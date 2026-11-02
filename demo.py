"""Offline demo: runs the ReAct agent on a multi-step task with no API key."""

from agent import ReActAgent
from llm import FakeLLM
from tools import TOOLS

TASK = "A car travels 120 km at a speed of 60 km/h. How long is the trip in hours, minutes, and seconds?"


def main():
    agent = ReActAgent(FakeLLM(), TOOLS)
    result = agent.run(TASK)
    print(f"Task: {TASK}\n")
    for i, step in enumerate(result.steps, 1):
        print(f"--- step {i} ---")
        print(step.thought)
        print(f"Observation: {step.observation}\n")
    print(f"Answer: {result.answer}")
    print(f"Converged: {result.converged}")


if __name__ == "__main__":
    main()
