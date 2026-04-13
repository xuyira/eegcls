from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class PreprocessConfig:
    sampling_rate: int
    window_size: int
    stride: int
    short_window_policy: str = "drop"
    per_window_normalize: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (window - mean) / std


def slice_windows(data: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, list[dict]]:
    """Slice a [C, T] array into windows shaped [N, C, T_window]."""
    num_channels, num_points = data.shape
    if num_points < config.window_size:
        if config.short_window_policy == "pad":
            padded = np.zeros((num_channels, config.window_size), dtype=np.float32)
            padded[:, :num_points] = data
            window = normalize_window(padded) if config.per_window_normalize else padded
            meta = [{"window_index": 0, "start_idx": 0, "end_idx": num_points}]
            return window[np.newaxis, ...], meta
        raise ValueError(
            f"Signal length {num_points} is shorter than window_size {config.window_size}"
        )

    windows = []
    meta = []
    starts = range(0, num_points - config.window_size + 1, config.stride)
    for window_index, start_idx in enumerate(starts):
        end_idx = start_idx + config.window_size
        window = data[:, start_idx:end_idx].astype(np.float32, copy=True)
        if config.per_window_normalize:
            window = normalize_window(window)
        windows.append(window)
        meta.append(
            {
                "window_index": window_index,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )
    if not windows:
        raise ValueError("No windows were produced from the input signal")
    return np.stack(windows, axis=0), meta
