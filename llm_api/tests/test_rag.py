import os
import sys
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Ensure llm_api directory (the one containing app.py) is importable from any pytest cwd.
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

app_module = importlib.import_module("app")


class FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [FakeEmbeddingItem(embedding) for embedding in embeddings]


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeEmbeddingsAPI:
    def create(self, model, input):
        def text_to_vec(text):
            lowered = text.lower()
            recursion_score = lowered.count("recursion") + lowered.count("base case")
            loop_score = lowered.count("loop") + lowered.count("iteration")
            return [float(recursion_score + 1), float(loop_score + 1)]

        if isinstance(input, list):
            return FakeEmbeddingResponse([text_to_vec(text) for text in input])
        return FakeEmbeddingResponse([text_to_vec(input)])


class FakeCompletionsAPI:
    def create(self, model, messages):
        return FakeChatResponse("Recursion is a function calling itself. [intro_cs_notes#1]")


class FakeChatAPI:
    def __init__(self):
        self.completions = FakeCompletionsAPI()


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsAPI()
        self.chat = FakeChatAPI()


@pytest.fixture(autouse=True)
def reset_state_and_client(monkeypatch):
    app_module.sessions.clear()
    app_module.chunks.clear()
    monkeypatch.setattr(app_module, "client", FakeClient())


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_chunking_overlap_correctness():
    chunks = app_module.split_text_into_chunks("abcdefghijklmnop", chunk_size=6, overlap=2)

    assert len(chunks) >= 2
    assert chunks[0][-2:] == chunks[1][:2]
    assert all(chunk.strip() for chunk in chunks)


def test_cosine_similarity_ranking():
    query = [1.0, 0.0]
    candidates = [
        {"id": "a", "vec": [1.0, 0.0]},
        {"id": "b", "vec": [0.0, 1.0]},
        {"id": "c", "vec": [0.5, 0.5]},
    ]

    scored = [
        {"id": c["id"], "score": app_module.cosine_similarity(query, c["vec"])}
        for c in candidates
    ]
    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)

    assert ranked[0]["id"] == "a"
    assert pytest.approx(ranked[0]["score"], rel=1e-6) == 1.0
    assert pytest.approx(ranked[-1]["score"], rel=1e-6) == 0.0


def test_ingest_and_search_workflow(client):
    text = (
        "Recursion solves problems by reducing them to smaller instances. "
        "A base case stops recursion. "
    ) * 20

    ingest_resp = client.post(
        "/ingest",
        json={"doc_id": "intro_cs_notes", "text": text},
    )
    assert ingest_resp.status_code == 200
    ingest_body = ingest_resp.json()
    assert ingest_body["doc_id"] == "intro_cs_notes"
    assert ingest_body["chunks_added"] > 0

    search_resp = client.get("/search", params={"query": "What is recursion?", "k": 3})
    assert search_resp.status_code == 200
    body = search_resp.json()
    assert body["query"] == "What is recursion?"
    assert len(body["results"]) >= 1
    assert set(body["results"][0].keys()) == {"chunk_id", "score", "text"}


def test_qa_endpoint_with_mocked_llm(client):
    session_resp = client.post("/session")
    assert session_resp.status_code == 200
    session_id = session_resp.json()["session_id"]

    ingest_resp = client.post(
        "/ingest",
        json={
            "doc_id": "intro_cs_notes",
            "text": "Recursion means a function calls itself. Base case ends recursion.",
        },
    )
    assert ingest_resp.status_code == 200

    qa_resp = client.post(
        "/qa",
        json={
            "session_id": session_id,
            "question": "What is recursion?",
            "k": 2,
        },
    )

    assert qa_resp.status_code == 200
    qa_body = qa_resp.json()
    assert "Recursion is a function calling itself" in qa_body["answer"]
    assert isinstance(qa_body["citations"], list)
    assert len(qa_body["citations"]) >= 1
    assert set(qa_body["citations"][0].keys()) == {"chunk_id", "score"}
    assert qa_body["turn_count"] == 1
