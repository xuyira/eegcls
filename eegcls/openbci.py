from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


HEADER_PREFIX = "EXG Channel"
NUM_CHANNELS = 8


def read_openbci_txt(path: str | Path, num_channels: int = NUM_CHANNELS) -> np.ndarray:
    """Read an OpenBCI txt file and return a [C, T] float32 array."""
    file_path = Path(path)
    rows: list[list[float]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header_seen = False
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            first_cell = row[0].strip()
            if not header_seen and first_cell.startswith(HEADER_PREFIX):
                header_seen = True
                continue
            if len(row) < num_channels:
                raise ValueError(
                    f"{file_path} row {row_idx + 1} has {len(row)} columns, expected at least {num_channels}"
                )
            try:
                rows.append([float(cell.strip()) for cell in row[:num_channels]])
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse numeric EEG values in {file_path} row {row_idx + 1}"
                ) from exc

    if not rows:
        raise ValueError(f"No EEG rows found in {file_path}")

    data = np.asarray(rows, dtype=np.float32)
    if data.shape[1] < num_channels:
        raise ValueError(f"{file_path} has only {data.shape[1]} EEG channels, expected {num_channels}")
    return data[:, :num_channels].T
