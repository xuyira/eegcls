from __future__ import annotations

import json
from pathlib import Path

import torch

from .modeling import build_model
from .preprocess import PreprocessConfig


def save_artifact(
    artifact_dir: str | Path,
    model: torch.nn.Module,
    model_config: dict,
    preprocess_config: dict,
    label_map: dict[str, int],
    train_summary: dict,
) -> Path:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (out_dir / "preprocess_config.json").write_text(json.dumps(preprocess_config, indent=2), encoding="utf-8")
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
    (out_dir / "library_version.txt").write_text("0.1.0\n", encoding="utf-8")
    return out_dir


def load_artifact(artifact_dir: str | Path) -> dict:
    path = Path(artifact_dir)
    model_config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
    preprocess_config_dict = json.loads((path / "preprocess_config.json").read_text(encoding="utf-8"))
    label_map = json.loads((path / "label_map.json").read_text(encoding="utf-8"))

    model = build_model(
        model_name=model_config["model_name"],
        num_channels=model_config["num_channels"],
        num_time=model_config["num_time"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
        sampling_rate=model_config.get("sampling_rate", preprocess_config_dict.get("sampling_rate", 250)),
    )
    try:
        state_dict = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "model_config": model_config,
        "preprocess_config": PreprocessConfig(**preprocess_config_dict),
        "label_map": label_map,
        "index_to_label": {index: label for label, index in label_map.items()},
    }
