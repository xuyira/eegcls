from __future__ import annotations

import argparse
import json

from eegcls.inference import predict_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-window inference on a single OpenBCI txt file.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    result = predict_file(args.artifact_dir, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
