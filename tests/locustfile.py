"""
Locust load testing script for LLM Playground API.

WHY THIS EXISTS:
Unit and integration tests verify correctness, but NOT performance under
load. You need to know:
- How many concurrent users can the API handle?
- What's the P99 latency under load?
- Where does the system break (CPU? GPU? memory? connections)?

USAGE:
    pip install locust
    locust -f tests/locustfile.py --host=http://localhost:8000

    Then open http://localhost:8089 to configure and run the test.
    OR run headless:
    locust -f tests/locustfile.py --host=http://localhost:8000 \
        --headless -u 100 -r 10 --run-time 60s
"""

import random

from locust import HttpUser, between, task


class LLMPlaygroundUser(HttpUser):
    """Simulates a user interacting with the LLM Playground API."""

    # Wait 1-3 seconds between requests (simulates human reading time)
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts. Could authenticate here."""
        # In DEV_MODE, no auth needed. In production, set up JWT here.
        pass

    @task(3)
    def chat_completion(self):
        """Most common action: send a chat message."""
        prompts = [
            "What is the capital of France?",
            "Explain how transformers work.",
            "Write a Python function to sort a list.",
            "What is gradient descent?",
            "Translate 'hello' to Spanish.",
        ]

        self.client.post(
            "/api/v1/chat/completions",
            json={
                "model": "llm-playground-7b",
                "messages": [
                    {"role": "user", "content": random.choice(prompts)}
                ],
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": False,
            },
            name="POST /chat/completions",
        )

    @task(1)
    def list_models(self):
        """Occasionally list available models."""
        self.client.get("/api/v1/models", name="GET /models")

    @task(1)
    def health_check(self):
        """Occasionally check health."""
        self.client.get("/health", name="GET /health")

    @task(1)
    def streaming_chat(self):
        """Test streaming (SSE) chat."""
        self.client.post(
            "/api/v1/chat/completions",
            json={
                "model": "llm-playground-7b",
                "messages": [
                    {"role": "user", "content": "Tell me a short joke."}
                ],
                "stream": True,
                "max_tokens": 50,
            },
            name="POST /chat/completions (streaming)",
            stream=True,  # Don't buffer the entire SSE response
        )


# ── Run headless example ──────────────────────────────────────────────
# locust -f tests/locustfile.py --host=http://localhost:8000 \
#     --headless -u 100 -r 10 --run-time 60s \
#     --html=load_test_report.html
#
# Parameters:
#   -u 100       = 100 concurrent users
#   -r 10        = spawn 10 users per second (ramp up)
#   --run-time   = stop after 60 seconds
#   --html       = save HTML report
