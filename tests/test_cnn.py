"""Architecture, real checkpoint inference, and upload validation."""

from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.cnn import AssignmentCNN, CLASSES
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as connection:
        yield connection


def test_specified_architecture_and_gradients():
    model = AssignmentCNN()
    x = torch.randn(2, 3, 64, 64)
    expected = [(2, 16, 64, 64), (2, 16, 64, 64), (2, 16, 32, 32),
                (2, 32, 32, 32), (2, 32, 32, 32), (2, 32, 16, 16),
                (2, 8192), (2, 100), (2, 100), (2, 10)]
    for layer, shape in zip(model.layers, expected):
        x = layer(x)
        assert tuple(x.shape) == shape
    assert sum(p.numel() for p in model.parameters()) == 825398
    torch.nn.functional.cross_entropy(x, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_uploaded_example_matches_direct_inference(client):
    path = Path(__file__).resolve().parent.parent / "examples" / "cifar10_test_0000.png"
    content = path.read_bytes()
    response = client.post("/classify", files={"file": (path.name, content, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body == app.state.classifier.predict(content)
    assert body["class_name"] == CLASSES[body["class_id"]]
    assert len(body["probabilities"]) == 10
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["confidence"] == max(body["probabilities"].values())
    assert not app.state.classifier.model.training


@pytest.mark.parametrize("mode", ["L", "RGBA", "RGB"])
def test_converts_image_modes_and_sizes(client, mode):
    stream = BytesIO()
    Image.new(mode, (47, 83)).save(stream, format="PNG")
    result = client.post("/classify", files={"file": ("image.png", stream.getvalue(), "image/png")})
    assert result.status_code == 200


@pytest.mark.parametrize("content,status", [(b"", 422), (b"not an image", 422),
                                           (b"x" * (5 * 1024 * 1024 + 1), 413)])
def test_rejects_invalid_uploads(client, content, status):
    assert client.post("/classify", files={"file": ("bad.png", content)}).status_code == status


def test_missing_file(client):
    assert client.post("/classify").status_code == 422
