#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator


N = 40960
D = 393
DATA_SEED = 550
PROTOTYPE_COUNT = 512
BASE_PROTOTYPE_COUNT = 256
REPLICAS_PER_PROTOTYPE = 80
OUTPUT_NAME = "local_memory_boundary_n40960_d393.csv"

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MIX1 = 0xBF58476D1CE4E5B9
MIX2 = 0x94D049BB133111EB
DENOMINATOR = 2048
DECIMAL_SCALE = 10**11


def splitmix64(value: int) -> int:
    value = (value + GAMMA) & MASK64
    mixed = value
    mixed = ((mixed ^ (mixed >> 30)) * MIX1) & MASK64
    mixed = ((mixed ^ (mixed >> 27)) * MIX2) & MASK64
    return (mixed ^ (mixed >> 31)) & MASK64


def prototype_odd(base_mode: int, dimension: int, data_seed: int) -> int:
    key = (
        data_seed
        ^ (((base_mode + 1) * GAMMA) & MASK64)
        ^ (((dimension + 1) * MIX1) & MASK64)
    )
    return 2 * ((splitmix64(key) >> 58) & 0x3F) - 63


def noise_odd(base_mode: int, replica: int, dimension: int, data_seed: int) -> int:
    if replica == 0:
        return 0
    key = (
        data_seed
        ^ (((base_mode + 1) * GAMMA) & MASK64)
        ^ (((replica + 1) * MIX1) & MASK64)
        ^ (((dimension + 1) * MIX2) & MASK64)
    )
    return 2 * ((splitmix64(key) >> 60) & 0xF) - 15


def numerator(mode: int, replica: int, dimension: int, data_seed: int) -> int:
    base_mode = mode % BASE_PROTOTYPE_COUNT
    sign = 1 if mode < BASE_PROTOTYPE_COUNT else -1
    return sign * (
        prototype_odd(base_mode, dimension, data_seed) * 256
        + noise_odd(base_mode, replica, dimension, data_seed)
    )


def decimal_text(value: int) -> str:
    sign = "-" if value < 0 else ""
    whole, remainder = divmod(abs(value), DENOMINATOR)
    fractional = remainder * DECIMAL_SCALE // DENOMINATOR
    return f"{sign}{whole}.{fractional:011d}"


def rows(n: int, d: int, data_seed: int) -> Iterator[tuple[int, ...]]:
    for point in range(n):
        replica, mode = divmod(point, PROTOTYPE_COUNT)
        yield tuple(numerator(mode, replica, dim, data_seed) for dim in range(d))


def generate(output_dir: Path) -> Path:
    if N != PROTOTYPE_COUNT * REPLICAS_PER_PROTOTYPE:
        raise ValueError("dataset size does not match the prototype layout")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_NAME
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite: {csv_path}")

    with csv_path.open("w", encoding="ascii", newline="\n") as csv_handle:
        csv_handle.write(",".join(f"f{index}" for index in range(D)) + "\n")
        for row in rows(N, D, DATA_SEED):
            csv_handle.write(",".join(decimal_text(value) for value in row) + "\n")

    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        csv_path = generate(args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Generated {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
