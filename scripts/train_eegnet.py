from __future__ import annotations

import argparse
import json

from eegcls.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EEG model on an OpenBCI dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--model-name", default="EEGNet")
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    summary = train(
        TrainConfig(
            dataset_root=args.dataset_root,
            artifact_dir=args.artifact_dir,
            model_name=args.model_name,
            sampling_rate=args.sampling_rate,
            window_size=args.window_size,
            stride=args.stride,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
