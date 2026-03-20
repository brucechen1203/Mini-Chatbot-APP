import os
import sys
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("OPENAI_API_KEY", "test-key")

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

app_module = importlib.import_module("app")


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletionsAPI:
    def create(self, model, messages):
        user_text = messages[-1]["content"]

        if "Return JSON only" in user_text:
            query_line = user_text.split("User query:", 1)[-1].split("\n", 1)[0].strip().lower()
            current_steps = user_text.split("Current steps:", 1)[-1].strip()

            if "[]" in current_steps:
                if any(token in query_line for token in ["*", "+", "-", "/"]) and any(ch.isdigit() for ch in query_line):
                    return FakeChatResponse('{"action":"calculator","input":"25*48"}')
                if any(keyword in query_line for keyword in ["recursion", "explain", "what is", "define"]):
                    return FakeChatResponse('{"action":"knowledge_base","input":"What is recursion?"}')
                return FakeChatResponse('{"action":"final","input":"Answer directly"}')

            if "calculator" in current_steps and "knowledge_base" not in current_steps and "recursion" in query_line:
                return FakeChatResponse('{"action":"knowledge_base","input":"explain recursion"}')

            return FakeChatResponse('{"action":"final","input":"Answer now"}')

        if "Tool steps" in user_text:
            if "1200" in user_text and "knowledge_base" in user_text:
                return FakeChatResponse("25 * 48 = 1200. Recursion is when a function calls itself with a base case.")
            if "1200" in user_text:
                return FakeChatResponse("25 * 48 = 1200.")
            if "knowledge_base" in user_text:
                return FakeChatResponse("Recursion is when a function calls itself. [intro_cs_notes#1]")
            return FakeChatResponse("This can be answered directly without tools.")

        return FakeChatResponse("Fallback response")


class FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [FakeEmbeddingItem(embedding) for embedding in embeddings]


class FakeEmbeddingsAPI:
    def create(self, model, input):
        def text_to_vec(text):
            lowered = text.lower()
            recursion_score = lowered.count("recursion") + lowered.count("base case")
            math_score = lowered.count("25") + lowered.count("48") + lowered.count("*")
            return [float(recursion_score + 1), float(math_score + 1)]

        if isinstance(input, list):
            return FakeEmbeddingResponse([text_to_vec(text) for text in input])
        return FakeEmbeddingResponse([text_to_vec(input)])


class FakeChatAPI:
    def __init__(self):
        self.completions = FakeCompletionsAPI()


class FakeClient:
    def __init__(self):
        self.chat = FakeChatAPI()
        self.embeddings = FakeEmbeddingsAPI()


@pytest.fixture(autouse=True)
def reset_state_and_client(monkeypatch):
    app_module.sessions.clear()
    app_module.chunks.clear()
    monkeypatch.setattr(app_module, "client", FakeClient())


@pytest.fixture
def client():
    return TestClient(app_module.app)


def create_session_id(client):
    response = client.post("/session")
    assert response.status_code == 200
    return response.json()["session_id"]


def test_agent_math_query_uses_calculator(client):
    session_id = create_session_id(client)

    response = client.post(
        "/agent",
        json={"session_id": session_id, "query": "What is 25*48?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "1200" in body["answer"]
    assert len(body["steps"]) >= 1
    assert body["steps"][0]["action"] == "calculator"
    assert body["steps"][0]["output"] == "1200"


def test_agent_knowledge_query_uses_rag(client):
    session_id = create_session_id(client)

    ingest = client.post(
        "/ingest",
        json={
            "doc_id": "intro_cs_notes",
            "text": "Recursion means a function calls itself. A base case stops recursion.",
        },
    )
    assert ingest.status_code == 200

    response = client.post(
        "/agent",
        json={"session_id": session_id, "query": "Explain recursion"},
    )

    assert response.status_code == 200
    body = response.json()
    assert any(step["action"] == "knowledge_base" for step in body["steps"])
    assert "Recursion" in body["answer"]


def test_agent_mixed_query_uses_multiple_tools(client):
    session_id = create_session_id(client)

    ingest = client.post(
        "/ingest",
        json={
            "doc_id": "intro_cs_notes",
            "text": "Recursion means a function calls itself. A base case stops recursion.",
        },
    )
    assert ingest.status_code == 200

    response = client.post(
        "/agent",
        json={"session_id": session_id, "query": "What is 25 * 48 and explain recursion?"},
    )

    assert response.status_code == 200
    body = response.json()
    actions = [step["action"] for step in body["steps"]]
    assert "calculator" in actions
    assert "knowledge_base" in actions
    assert "1200" in body["answer"]


def test_agent_simple_query_avoids_tools(client):
    session_id = create_session_id(client)

    response = client.post(
        "/agent",
        json={"session_id": session_id, "query": "Say hello"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["steps"] == []
    assert "directly" in body["answer"].lower()


def test_safe_calculator_rejects_unsafe_input():
    with pytest.raises(ValueError):
        app_module.safe_calculator("__import__('os').system('echo hi')")
