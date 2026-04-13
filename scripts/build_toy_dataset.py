from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPLITS = ("train", "val", "test")
LABELS = ("class0", "class1", "class2")


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"Empty file: {path}")
    return rows[0], rows[1:]


def write_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a toy 3-class dataset from one OpenBCI txt file.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source)
    out_root = Path(args.output)
    header, rows = read_rows(source_path)
    total_chunks = len(SPLITS) * len(LABELS)
    chunk_size = len(rows) // total_chunks
    if chunk_size < 128:
        raise ValueError(
            f"Source file is too short for a useful toy dataset: {len(rows)} rows across {total_chunks} chunks"
        )

    for split_idx, split in enumerate(SPLITS):
        for label_idx, label in enumerate(LABELS):
            chunk_index = split_idx * len(LABELS) + label_idx
            start = chunk_index * chunk_size
            end = (chunk_index + 1) * chunk_size if chunk_index < total_chunks - 1 else len(rows)
            chunk_rows = rows[start:end]
            dest = out_root / split / label / f"{source_path.stem}_{split}_{label}.txt"
            write_rows(dest, header, chunk_rows)
            print(f"Wrote {dest} with {len(chunk_rows)} rows")


if __name__ == "__main__":
    main()
