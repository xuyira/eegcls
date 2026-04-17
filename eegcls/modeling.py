from __future__ import annotations

from pathlib import Path
import sys

import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.EEGDeformer import Deformer  # noqa: E402
from models.EEGNet import eegNet  # noqa: E402
from models.EEGViT import EEGViT  # noqa: E402
from models.LGGNet import LGGNet  # noqa: E402
from models.TSception import TSception  # noqa: E402
from models.conformer import Conformer  # noqa: E402


SUPPORTED_MODELS = ("EEGNet", "TSception", "EEGViT", "Conformer", "Deformer", "LGGNet")


def _effective_sampling_rate(num_time: int, sampling_rate: int) -> int:
    return max(32, min(sampling_rate, num_time))


def _choose_num_patches(num_time: int) -> int:
    for candidate in (40, 32, 24, 20, 16, 12, 10, 8, 5, 4, 2, 1):
        if num_time % candidate == 0:
            return candidate
    return 1


def _estimate_conformer_hidden(num_time: int) -> int:
    patch_tokens = max(((num_time - 100) // 15) + 1, 1)
    return patch_tokens * 40


def _default_idx_graph(num_channels: int) -> list[int]:
    base = num_channels // 4
    remainder = num_channels % 4
    groups = [base] * 4
    for index in range(remainder):
        groups[index] += 1
    return [group for group in groups if group > 0]


def build_model(
    model_name: str,
    num_channels: int,
    num_time: int,
    num_classes: int,
    dropout: float,
    sampling_rate: int = 250,
) -> nn.Module:
    if model_name == "EEGNet":
        return eegNet(
            nChan=num_channels,
            nTime=num_time,
            nClass=num_classes,
            dropoutP=dropout,
            F1=8,
            D=2,
            C1=64,
        )

    if model_name == "TSception":
        return TSception(
            num_classes=num_classes,
            input_size=(1, num_channels, num_time),
            sampling_rate=_effective_sampling_rate(num_time, sampling_rate),
            num_T=15,
            num_S=15,
            hidden=32,
            dropout_rate=dropout,
        )

    if model_name == "EEGViT":
        return EEGViT(
            num_chan=num_channels,
            num_time=num_time,
            num_patches=_choose_num_patches(num_time),
            num_classes=num_classes,
            dropout=dropout,
            emb_dropout=dropout,
        )

    if model_name == "Conformer":
        return Conformer(
            n_classes=num_classes,
            n_chan=num_channels,
            n_hidden=_estimate_conformer_hidden(num_time),
        )

    if model_name == "Deformer":
        return Deformer(
            num_chan=num_channels,
            num_time=num_time,
            temporal_kernel=min(11, num_time if num_time % 2 == 1 else max(num_time - 1, 1)),
            num_kernel=32,
            num_classes=num_classes,
            depth=2,
            heads=4,
            mlp_dim=16,
            dim_head=8,
            dropout=dropout,
        )

    if model_name == "LGGNet":
        effective_sampling_rate = _effective_sampling_rate(num_time, sampling_rate)
        return LGGNet(
            num_classes=num_classes,
            input_size=(1, num_channels, num_time),
            sampling_rate=effective_sampling_rate,
            num_T=8,
            out_graph=16,
            dropout_rate=dropout,
            pool=max(4, min(8, num_time // 16)),
            pool_step_rate=0.25,
            idx_graph=_default_idx_graph(num_channels),
        )

    raise ValueError(
        f"Unsupported model_name: {model_name}. Supported models: {', '.join(SUPPORTED_MODELS)}"
    )
