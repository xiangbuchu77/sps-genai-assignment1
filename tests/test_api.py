"""Integration checks using the actual course model, without mocked vectors."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.bigram_model import BigramModel


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_root(client):
    assert client.get("/").json() == {"Hello": "World"}


@pytest.mark.parametrize("word", ["apple", "king", "car"])
def test_embedding_matches_course_model(client, word):
    response = client.post("/embedding", json={"word": word})
    assert response.status_code == 200
    data = response.json()
    assert data["dimensions"] == len(data["embedding"]) == 300
    expected = app.state.nlp(word).vector
    np.testing.assert_allclose(data["embedding"], expected, rtol=0, atol=0)
    assert np.isfinite(data["embedding"]).all()
    assert np.linalg.norm(data["embedding"]) > 0


def test_trims_whitespace(client):
    response = client.post("/embedding", json={"word": "  apple  "})
    assert response.status_code == 200
    assert response.json()["word"] == "apple"


@pytest.mark.parametrize("payload", [{}, {"word": ""}, {"word": "   "}, {"word": 123},
                                     {"word": "two words"}, {"word": "!"}, {"word": "a" * 101}])
def test_invalid_embedding_requests(client, payload):
    assert client.post("/embedding", json=payload).status_code == 422


def test_unknown_word(client):
    assert client.post("/embedding", json={"word": "zzqzxqvnotawordzz"}).status_code == 404


def test_generation(client):
    response = client.post("/generate", json={"start_word": "the", "length": 8})
    assert response.status_code == 200
    words = response.json()["generated_text"].split()
    assert words[0] == "the"
    assert 1 <= len(words) <= 8


@pytest.mark.parametrize("length", [0, -1, 201, 1.5, True])
def test_invalid_generation_length(client, length):
    assert client.post("/generate", json={"start_word": "the", "length": length}).status_code == 422


def test_bigram_dead_end_and_boundary():
    model = BigramModel(["a b c", "d e"])
    assert model.generate_text("a", 10) == "a b c"
    assert model.generate_text("unknown", 10) == "unknown"
    assert model.generate_text("a", 1) == "a"
