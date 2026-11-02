"""Tests for tools and the ReAct loop (all offline, no API key)."""

import unittest

from agent import ReActAgent
from llm import FakeLLM
from tools import TOOLS, calculator, unit_convert


class TestTools(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculator("120/60"), "2.0")
        self.assertEqual(calculator("sqrt(81)"), "9.0")
        self.assertEqual(calculator("2**10"), "1024")

    def test_calculator_rejects_unsafe(self):
        self.assertTrue(calculator("__import__('os').system('x')").startswith("Error"))

    def test_unit_convert(self):
        self.assertEqual(unit_convert("2.5,km,mi"), str(2.5 * 1000 / 1609.344))
        self.assertEqual(unit_convert("120,minutes,seconds"), "7200.0")
        self.assertTrue(unit_convert("1,km,hours").startswith("Error"))


class TestAgentLoop(unittest.TestCase):
    def test_demo_task_converges(self):
        agent = ReActAgent(FakeLLM(), TOOLS)
        result = agent.run("A car travels 120 km at 60 km/h. How long in hours, minutes, seconds?")
        self.assertTrue(result.converged)
        self.assertIn("7200", result.answer)
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.steps[0].action, "calculator")

    def test_unknown_tool_surfaced_as_observation(self):
        class ConfusedLLM:
            def chat(self, messages):
                n = sum(1 for m in messages if m["role"] == "assistant")
                if n == 0:
                    return "Thought: trying something.\nAction: nope[1]"
                return "Thought: I now know the final answer.\nFinal Answer: done"

        agent = ReActAgent(ConfusedLLM(), TOOLS)
        result = agent.run("dummy")
        self.assertTrue(result.converged)
        self.assertIn("unknown tool", result.steps[0].observation)


if __name__ == "__main__":
    unittest.main()
