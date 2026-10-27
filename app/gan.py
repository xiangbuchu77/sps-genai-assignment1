"""The 28x28 unconditional GAN architecture specified in Assignment 3."""

import torch
from torch import nn

LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(LATENT_DIM, 128 * 7 * 7),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        if noise.ndim != 2 or noise.shape[1] != LATENT_DIM:
            raise ValueError("Generator expects noise with shape [B, 100].")
        return self.layers(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("Discriminator expects images with shape [B, 1, 28, 28].")
        # BCEWithLogitsLoss includes sigmoid during training. sigmoid(logit)
        # gives the real/fake probability described in the assignment.
        return self.layers(images)


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def train_step(generator, discriminator, real, optimizer_g, optimizer_d):
    """One discriminator update followed by one non-saturating generator update."""
    criterion = nn.BCEWithLogitsLoss()
    batch_size = len(real)
    real_targets = real.new_ones(batch_size, 1)
    fake_targets = real.new_zeros(batch_size, 1)
    noise = torch.randn(batch_size, LATENT_DIM, device=real.device)

    optimizer_d.zero_grad(set_to_none=True)
    fake = generator(noise)
    real_logits = discriminator(real)
    fake_logits = discriminator(fake.detach())
    loss_d = criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)
    if not torch.isfinite(loss_d):
        raise FloatingPointError("Discriminator loss is not finite.")
    loss_d.backward()
    optimizer_d.step()

    # Backpropagate through D to G without accumulating gradients on D.
    optimizer_g.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in discriminator.parameters()]
    try:
        discriminator.requires_grad_(False)
        loss_g = criterion(discriminator(fake), real_targets)
        if not torch.isfinite(loss_g):
            raise FloatingPointError("Generator loss is not finite.")
        loss_g.backward()
        optimizer_g.step()
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags):
            parameter.requires_grad_(flag)
    return {"discriminator_loss": loss_d.item(), "generator_loss": loss_g.item(),
            "real_probability": real_logits.detach().sigmoid().mean().item(),
            "fake_probability": fake_logits.detach().sigmoid().mean().item()}
