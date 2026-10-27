"""Assignment architecture, adversarial updates, image API and written calculations."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch

from app.gan import Generator, Discriminator, train_step
from app.digit_generator import DigitGenerator
from app.main import app

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as connection:
        yield connection


def test_required_architecture_and_gradients():
    generator, discriminator = Generator(), Discriminator()
    noise = torch.randn(3, 100)
    x = noise
    for layer, shape in zip(generator.layers, [(3, 6272), (3, 128, 7, 7),
            (3, 64, 14, 14), (3, 64, 14, 14), (3, 64, 14, 14),
            (3, 1, 28, 28), (3, 1, 28, 28)]):
        x = layer(x)
        assert tuple(x.shape) == shape
    assert -1 <= x.min() <= x.max() <= 1
    for layer, shape in zip(discriminator.layers, [(3, 64, 14, 14), (3, 64, 14, 14),
            (3, 128, 7, 7), (3, 128, 7, 7), (3, 128, 7, 7), (3, 6272), (3, 1)]):
        x = layer(x)
        assert tuple(x.shape) == shape
    torch.nn.functional.binary_cross_entropy_with_logits(x, torch.ones_like(x)).backward()
    for network in [generator, discriminator]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    assert sum(p.numel() for p in generator.parameters()) == 765761
    assert sum(p.numel() for p in discriminator.parameters()) == 138817


def test_training_updates_both_networks():
    generator, discriminator = Generator(), Discriminator()
    originals = [[p.clone() for p in network.parameters()] for network in [generator, discriminator]]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    metrics = train_step(generator, discriminator, torch.rand(5, 1, 28, 28) * 2 - 1,
                         optimizer_g, optimizer_d)
    for before, network in zip(originals, [generator, discriminator]):
        assert any(not torch.equal(a, b) for a, b in zip(before, network.parameters()))
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert all(p.requires_grad for p in discriminator.parameters())


def test_checkpoint_sampling_preserves_buffers_and_global_rng():
    service = DigitGenerator()
    before = {k: v.clone() for k, v in service.model.named_buffers()}
    rng_before = torch.random.get_rng_state().clone()
    images = service.sample(7, 1234)
    assert images.shape == (7, 1, 28, 28)
    assert torch.isfinite(images).all() and -1 <= images.min() <= images.max() <= 1
    assert torch.equal(images, service.sample(7, 1234))
    assert not torch.equal(images, service.sample(7, 4321))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in service.model.named_buffers())
    assert not service.model.training


def test_image_endpoint_matches_direct_generation(client):
    response = client.post("/generate-image", json={"num_samples": 9, "seed": 42})
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.headers["x-num-samples"] == "9" and response.headers["x-seed"] == "42"
    assert response.content == app.state.digit_generator.png(9, 42)
    with Image.open(BytesIO(response.content)) as image:
        assert image.mode == "L" and image.size == (92, 92)


@pytest.mark.parametrize("payload", [
    {"num_samples": 0}, {"num_samples": 65}, {"num_samples": 1.5},
    {"num_samples": True}, {"num_samples": "4"},
    {"seed": -1}, {"seed": 2**32}, {"seed": 0.5}, {"seed": True},
])
def test_invalid_generation_parameters(client, payload):
    assert client.post("/generate-image", json=payload).status_code == 422


def test_previous_text_and_embedding_routes(client):
    assert client.get("/").json() == {"Hello": "World"}
    generated = client.post("/generate", json={"start_word": "the", "length": 8})
    assert generated.status_code == 200
    assert generated.json()["generated_text"].split()[0] == "the"
    embedding = client.post("/embedding", json={"word": "apple"})
    assert embedding.status_code == 200 and embedding.json()["dimensions"] == 300


def test_transposed_convolution_calculations():
    for stride, expected in [(2, 17), (3, 24)]:
        layer = torch.nn.ConvTranspose2d(1, 1, 4, stride=stride, padding=1, output_padding=1)
        assert layer(torch.zeros(1, 1, 8, 8)).shape[-2:] == (expected, expected)
    layer = torch.nn.ConvTranspose2d(1, 1, 2, stride=2)
    assert layer(torch.zeros(1, 1, 16, 16)).shape[-2:] == (32, 32)


def test_batchnorm_hand_calculation():
    values = torch.tensor([6., 8., 10., 6.], dtype=torch.float64)
    normalized = (values - values.mean()) / values.var(correction=0).sqrt()
    assert values.mean().item() == 7.5 and values.var(correction=0).item() == 2.75
    torch.testing.assert_close(normalized, torch.tensor([-0.9045340337, 0.3015113446,
                                                        1.5075567229, -0.9045340337], dtype=torch.float64))
    actual = torch.nn.functional.batch_norm(values[:, None], None, None, training=True, eps=1e-12)
    torch.testing.assert_close(normalized, actual.flatten(), rtol=1e-10, atol=1e-10)
