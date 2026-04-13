from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .openbci import read_openbci_txt
from .preprocess import PreprocessConfig, slice_windows


@dataclass(frozen=True)
class WindowRecord:
    path: Path
    label_name: str
    label_index: int
    split: str
    window_index: int
    start_idx: int
    end_idx: int


def build_label_map(dataset_root: str | Path) -> dict[str, int]:
    root = Path(dataset_root)
    labels = sorted(
        {path.name for split in ("train", "val", "test") for path in (root / split).iterdir() if path.is_dir()}
    )
    if not labels:
        raise ValueError(f"No label directories found under {root}")
    return {label: idx for idx, label in enumerate(labels)}


class OpenBCIWindowDataset(Dataset):
    def __init__(self, dataset_root: str | Path, split: str, preprocess_config: PreprocessConfig):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.preprocess_config = preprocess_config
        self.label_map = build_label_map(self.dataset_root)
        self.samples: list[np.ndarray] = []
        self.targets: list[int] = []
        self.records: list[WindowRecord] = []
        self._load()

    def _load(self) -> None:
        split_root = self.dataset_root / self.split
        if not split_root.exists():
            raise ValueError(f"Missing split directory: {split_root}")

        for label_name, label_index in self.label_map.items():
            label_dir = split_root / label_name
            if not label_dir.exists():
                continue
            for txt_path in sorted(label_dir.glob("*.txt")):
                signal = read_openbci_txt(txt_path)
                windows, meta = slice_windows(signal, self.preprocess_config)
                for window, item in zip(windows, meta):
                    self.samples.append(window)
                    self.targets.append(label_index)
                    self.records.append(
                        WindowRecord(
                            path=txt_path,
                            label_name=label_name,
                            label_index=label_index,
                            split=self.split,
                            window_index=item["window_index"],
                            start_idx=item["start_idx"],
                            end_idx=item["end_idx"],
                        )
                    )

        if not self.samples:
            raise ValueError(f"No samples loaded from {split_root}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.samples[index]).float()
        y = torch.tensor(self.targets[index], dtype=torch.long)
        return x, y
