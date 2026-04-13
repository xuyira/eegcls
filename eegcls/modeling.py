from __future__ import annotations

from pathlib import Path
import sys

import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.EEGNet import eegNet  # noqa: E402


def build_model(model_name: str, num_channels: int, num_time: int, num_classes: int, dropout: float) -> nn.Module:
    if model_name != "EEGNet":
        raise ValueError(f"Unsupported model_name: {model_name}")
    return eegNet(
        nChan=num_channels,
        nTime=num_time,
        nClass=num_classes,
        dropoutP=dropout,
        F1=8,
        D=2,
        C1=64,
    )
