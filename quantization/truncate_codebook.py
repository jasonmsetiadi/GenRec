"""Create a collision-free variable-length SID codebook from fixed-length IDs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def truncate_codebook(
    codebook: np.ndarray,
    min_length: int,
    max_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if codebook.ndim != 2:
        raise ValueError(f"Expected a rectangular 2D codebook, got shape {codebook.shape}.")
    if not 1 <= min_length <= max_length <= codebook.shape[1]:
        raise ValueError(
            f"Lengths must satisfy 1 <= min_length <= max_length <= {codebook.shape[1]}."
        )

    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_length, max_length + 1, size=codebook.shape[0])

    while True:
        prefixes = defaultdict(list)
        for index, length in enumerate(lengths):
            prefixes[tuple(codebook[index, :length].tolist())].append(index)

        collisions = [indices for indices in prefixes.values() if len(indices) > 1]
        if not collisions:
            break

        changed = False
        for indices in collisions:
            for index in indices:
                if lengths[index] < max_length:
                    lengths[index] += 1
                    changed = True
        if not changed:
            raise ValueError(
                "Fixed-length codebook contains duplicate IDs within the selected "
                "maximum length, so collision-free truncation is impossible."
            )

    ragged = np.empty(codebook.shape[0], dtype=object)
    for index, length in enumerate(lengths):
        ragged[index] = codebook[index, :length].astype(np.int64, copy=True)
    return ragged, lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly truncate fixed SIDs into collision-free variable-length SIDs."
    )
    parser.add_argument("--input", required=True, help="Input fixed-length .npy codebook.")
    parser.add_argument("--output", required=True, help="Output ragged object-array .npy codebook.")
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    codebook = np.load(input_path, allow_pickle=False)
    ragged, lengths = truncate_codebook(
        codebook,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, ragged, allow_pickle=True)
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "seed": args.seed,
                "items": int(len(ragged)),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "mean_length": float(lengths.mean()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(ragged)} collision-free variable-length SIDs to {output_path}")
    print(f"Saved length summary to {summary_path}")


if __name__ == "__main__":
    main()
