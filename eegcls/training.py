from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifact import save_artifact
from .dataset import OpenBCIWindowDataset
from .modeling import build_model
from .preprocess import PreprocessConfig


@dataclass
class TrainConfig:
    dataset_root: str
    artifact_dir: str
    sampling_rate: int = 250
    window_size: int = 128
    stride: int = 64
    short_window_policy: str = "drop"
    model_name: str = "EEGNet"
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 1e-3
    dropout: float = 0.25
    seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def train(config: TrainConfig) -> dict:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = PreprocessConfig(
        sampling_rate=config.sampling_rate,
        window_size=config.window_size,
        stride=config.stride,
        short_window_policy=config.short_window_policy,
    )

    train_ds = OpenBCIWindowDataset(config.dataset_root, "train", preprocess_config)
    val_ds = OpenBCIWindowDataset(config.dataset_root, "val", preprocess_config)
    test_ds = OpenBCIWindowDataset(config.dataset_root, "test", preprocess_config)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = build_model(
        model_name=config.model_name,
        num_channels=8,
        num_time=config.window_size,
        num_classes=len(train_ds.label_map),
        dropout=config.dropout,
        sampling_rate=config.sampling_rate,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_state = None
    best_val_acc = -1.0
    history = []
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_metrics = {"loss": running_loss / max(seen, 1)}
        val_metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    model_config = {
        "model_name": config.model_name,
        "num_channels": 8,
        "num_time": config.window_size,
        "num_classes": len(train_ds.label_map),
        "dropout": config.dropout,
        "sampling_rate": config.sampling_rate,
    }
    train_summary = {
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "history": history,
        "train_config": config.to_dict(),
    }

    save_artifact(
        artifact_dir=config.artifact_dir,
        model=model.cpu(),
        model_config=model_config,
        preprocess_config=preprocess_config.to_dict(),
        label_map=train_ds.label_map,
        train_summary=train_summary,
    )
    return train_summary
