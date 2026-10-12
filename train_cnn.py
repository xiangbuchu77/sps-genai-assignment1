"""Train on CIFAR-10; select on validation data and evaluate the test set once."""

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
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from app.cnn import AssignmentCNN, CLASSES, image_transform


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, correct, count = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(1)
            loss_sum += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            count += labels.size(0)
            if optimizer is None:
                ids = (labels * 10 + predicted).cpu()
                confusion += torch.bincount(ids, minlength=100).reshape(10, 10)
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_data = CIFAR10(args.data_dir, train=True, download=True, transform=image_transform(True))
    evaluation_data = CIFAR10(args.data_dir, train=True, download=False, transform=image_transform())
    # Save the actual indices as well as the seed so the split is inspectable.
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_ids, val_ids = indices[:45000], indices[45000:]
    (output / "split_indices.json").write_text(json.dumps({"train": train_ids, "validation": val_ids}))
    generator = torch.Generator().manual_seed(args.seed)
    options = {"batch_size": args.batch_size, "num_workers": args.workers,
               "persistent_workers": args.workers > 0}
    train_loader = DataLoader(Subset(training_data, train_ids), shuffle=True, generator=generator, **options)
    val_loader = DataLoader(Subset(evaluation_data, val_ids), shuffle=False, **options)
    model = AssignmentCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, best_accuracy, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    print(f"Device: {device}; train=45000, validation=5000, test=10000", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train, _ = run_epoch(model, train_loader, device, optimizer)
        validation, _ = run_epoch(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": learning_rate, "train": train,
               "validation": validation, "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "classes": list(CLASSES), "image_size": 64, "epoch": epoch,
                        "validation_accuracy": best_accuracy}, output / "cifar10_cnn.pt")
        (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"Epoch {epoch:02d}/{args.epochs}: train loss={train['loss']:.4f} "
              f"acc={train['accuracy']:.2%}; val loss={validation['loss']:.4f} "
              f"acc={validation['accuracy']:.2%}; {row['seconds']:.1f}s", flush=True)
    checkpoint = torch.load(output / "cifar10_cnn.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_data = CIFAR10(args.data_dir, train=False, download=False, transform=image_transform())
    test, confusion = run_epoch(model, DataLoader(test_data, shuffle=False, **options), device)
    metrics = {"dataset": "CIFAR-10", "seed": args.seed, "device": str(device),
               "python": platform.python_version(), "torch": torch.__version__,
               "torchvision": torchvision.__version__, "epochs": args.epochs,
               "batch_size": args.batch_size, "optimizer": "AdamW", "initial_lr": 0.001,
               "weight_decay": 0.0001, "scheduler": "CosineAnnealingLR",
               "train_count": 45000, "validation_count": 5000,
               "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
               "test": test, "classes": list(CLASSES),
               "confusion_matrix": confusion.tolist(),
               "per_class_accuracy": {c: confusion[i, i].item() / confusion[i].sum().item()
                                      for i, c in enumerate(CLASSES)},
               "parameters": sum(p.numel() for p in model.parameters()),
               "elapsed_seconds": time.perf_counter() - started,
               "checkpoint_sha256": hashlib.sha256((output / "cifar10_cnn.pt").read_bytes()).hexdigest()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    examples = Path("examples")
    examples.mkdir(exist_ok=True)
    # Fixed test index, chosen independently of its prediction.
    image, label = CIFAR10(args.data_dir, train=False)[0]
    image.save(examples / "cifar10_test_0000.png")
    print(f"Best epoch: {best_epoch}; test accuracy: {test['accuracy']:.2%}; test loss: {test['loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
