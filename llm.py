"""LLM backends: an OpenAI-compatible HTTP client (stdlib) and an offline stand-in."""

import json
import os
import urllib.request


class OpenAICompatibleLLM:
    """Minimal chat client for any OpenAI-compatible API. Needs OPENAI_API_KEY."""

    def __init__(self, model="gpt-4o-mini", base_url=None, api_key=None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL",
                                                     "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY (or pass api_key explicitly).")

    def chat(self, messages):
        payload = json.dumps({"model": self.model, "messages": messages,
                              "temperature": 0}).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"]


class FakeLLM:
    """Deterministic scripted LLM so the demo and tests run with no API key.

    Plays a fixed ReAct script for the demo task (trip-time arithmetic):
    three calculator calls, then the final answer.
    """

    _SCRIPT = [
        "Thought: I need the travel time in hours first: 120 km at 60 km/h.\nAction: calculator[120/60]",
        "Thought: That's 2.0 hours. Now convert hours to minutes.\nAction: calculator[2.0*60]",
        "Thought: 120.0 minutes. Now convert minutes to seconds.\nAction: calculator[120.0*60]",
        ("Thought: I now know the final answer.\nFinal Answer: The trip takes 2.0 hours, "
         "which is 120.0 minutes or 7200.0 seconds."),
    ]

    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        reply = self._SCRIPT[min(self.calls, len(self._SCRIPT) - 1)]
        self.calls += 1
        return reply
