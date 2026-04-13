from __future__ import annotations

from pathlib import Path

import torch

from .artifact import load_artifact
from .openbci import read_openbci_txt
from .preprocess import slice_windows


@torch.no_grad()
def predict_file(artifact_dir: str | Path, txt_path: str | Path) -> list[dict]:
    artifact = load_artifact(artifact_dir)
    signal = read_openbci_txt(txt_path)
    windows, meta = slice_windows(signal, artifact["preprocess_config"])
    x = torch.from_numpy(windows).float()
    logits = artifact["model"](x)
    probabilities = torch.softmax(logits, dim=1)
    confidence, pred_index = torch.max(probabilities, dim=1)

    sampling_rate = artifact["preprocess_config"].sampling_rate
    results = []
    for idx, item in enumerate(meta):
        results.append(
            {
                "window_index": item["window_index"],
                "start_idx": item["start_idx"],
                "end_idx": item["end_idx"],
                "start_time": item["start_idx"] / sampling_rate,
                "end_time": item["end_idx"] / sampling_rate,
                "pred_index": int(pred_index[idx].item()),
                "pred_label": artifact["index_to_label"][int(pred_index[idx].item())],
                "confidence": float(confidence[idx].item()),
                "probabilities": probabilities[idx].tolist(),
            }
        )
    return results
