"""Load the trained MNIST generator and return a reproducible PNG grid."""

from io import BytesIO
import math
from pathlib import Path

from PIL import Image
import torch
from torchvision.utils import make_grid

from app.gan import Generator, LATENT_DIM

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models/mnist_gan.pt"


def image_grid(images):
    """Convert raw Tanh images to a grayscale PNG-compatible PIL image."""
    display = (images.detach().cpu() + 1).div(2).clamp(0, 1)
    grid = make_grid(display, nrow=math.ceil(math.sqrt(len(display))), padding=2)
    pixels = grid[0].mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(pixels)


class DigitGenerator:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["latent_dim"] != LATENT_DIM or checkpoint["image_shape"] != [1, 28, 28]:
            raise ValueError("Incompatible MNIST generator checkpoint.")
        self.model = Generator()
        self.model.load_state_dict(checkpoint["generator_state_dict"])
        self.model.eval()
        self.epoch = checkpoint["epoch"]

    @torch.inference_mode()
    def sample(self, num_samples=16, seed=42):
        if not isinstance(num_samples, int) or not 1 <= num_samples <= 64:
            raise ValueError("num_samples must be an integer from 1 to 64.")
        if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer from 0 to 4294967295.")
        # Each request owns its RNG; sampling does not reset the global RNG.
        noise = torch.randn(num_samples, LATENT_DIM, generator=torch.Generator().manual_seed(seed))
        return self.model(noise)

    def png(self, num_samples=16, seed=42):
        stream = BytesIO()
        image_grid(self.sample(num_samples, seed)).save(stream, format="PNG")
        return stream.getvalue()
