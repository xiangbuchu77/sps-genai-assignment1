"""Train the required GAN on all 60,000 MNIST training images."""

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from app.gan import Generator, Discriminator, LATENT_DIM, initialize_weights, train_step
from app.digit_generator import image_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews = Path("tmp/assignment3/previews")
    previews.mkdir(parents=True, exist_ok=True)
    examples = Path("examples")
    examples.mkdir(exist_ok=True)

    dataset = MNIST(args.data_dir, train=True, download=True)
    # Exactly ToTensor + Normalize(0.5, 0.5), performed once for efficient loading.
    images = dataset.data.unsqueeze(1).float().div_(127.5).sub_(1)
    loader = DataLoader(TensorDataset(images), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, LATENT_DIM, generator=torch.Generator().manual_seed(1234)).to(device)
    history = []
    started = time.perf_counter()
    print(f"Device: {device}; MNIST training images: {len(dataset)}; epochs: {args.epochs}", flush=True)
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        epoch_start = time.perf_counter()
        sums = {"discriminator_loss": 0.0, "generator_loss": 0.0,
                "real_probability": 0.0, "fake_probability": 0.0}
        count = updates = 0
        for (batch,) in loader:
            batch = batch.to(device)
            result = train_step(generator, discriminator, batch, optimizer_g, optimizer_d)
            for key, value in result.items():
                sums[key] += value * len(batch)
            count += len(batch)
            updates += 1
        record = {"epoch": epoch, "examples": count, "updates_per_network": updates,
                  **{key: value / count for key, value in sums.items()},
                  "seconds": time.perf_counter() - epoch_start}
        history.append(record)
        (args.output_dir / "gan_training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        generator.eval()
        with torch.inference_mode():
            sample = generator(fixed_noise).cpu()
        image_grid(sample).resize((968, 968), resample=0).save(previews / f"epoch_{epoch:03d}.png")
        # Keep a checkpoint each epoch so completed work survives interruptions.
        checkpoint = {"generator_state_dict": {k: v.cpu() for k, v in generator.state_dict().items()},
                      "discriminator_state_dict": {k: v.cpu() for k, v in discriminator.state_dict().items()},
                      "latent_dim": LATENT_DIM, "image_shape": [1, 28, 28],
                      "epoch": epoch, "dataset": "MNIST", "train_count": len(dataset), "seed": args.seed}
        torch.save(checkpoint, args.output_dir / "mnist_gan.pt")
        print(f"Epoch {epoch:02d}/{args.epochs}: D={record['discriminator_loss']:.4f}, "
              f"G={record['generator_loss']:.4f}, {record['seconds']:.1f}s", flush=True)

    image_grid(sample).resize((968, 968), resample=0).save(examples / "mnist_generated.png")
    metrics = {"dataset": "MNIST", "split": "official training split", "train_count": len(dataset),
               "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
               "device": str(device), "python": platform.python_version(),
               "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
               "latent_dim": LATENT_DIM, "image_shape": [1, 28, 28],
               "generator_parameters": sum(p.numel() for p in generator.parameters()),
               "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
               "optimizer": "Adam", "learning_rate": 0.0002, "betas": [0.5, 0.999],
               "loss": "BCEWithLogitsLoss; non-saturating generator objective",
               "updates_per_network": sum(r["updates_per_network"] for r in history),
               "preview_seed": 1234, "preview_count": 64,
               "elapsed_seconds": time.perf_counter() - started,
               "final_epoch": history[-1],
               "checkpoint_sha256": hashlib.sha256((args.output_dir / "mnist_gan.pt").read_bytes()).hexdigest()}
    (args.output_dir / "gan_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
