#!/usr/bin/env python3
"""Guard public synthetic generator output compatibility without executing them."""

from __future__ import annotations

import argparse
from pathlib import Path


SCALABILITY_VARIANTS = {
    "n1e5": "25000, 25000, 25000, 25000",
    "n1e6": "250000, 250000, 250000, 250000",
    "n1e7": "2500000, 2500000, 2500000, 2500000",
    "n1e8": "25000000, 25000000, 25000000, 25000000",
}


def require(source: str, fragment: str, context: str) -> None:
    if fragment not in source:
        raise AssertionError(f"{context} contract missing: {fragment!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    guide_path = root / "Datasets/SYNTHETIC_DATASETS.md"
    guide = guide_path.read_text(encoding="utf-8")
    normalized_guide = " ".join(guide.split())

    for variant, cluster_sizes in SCALABILITY_VARIANTS.items():
        relative = Path("Datasets/scalability") / f"generate_scalability_{variant}.py"
        source = (root / relative).read_text(encoding="utf-8")
        context = relative.as_posix()
        output_open = f'file = open("scalability_{variant}.csv", "w")'
        header_write = 'file.write("dim1,dim2,dim3,dim4\\n")'
        require(source, output_open, context)
        require(source, header_write, context)
        require(source, f"nb_points = [{cluster_sizes}]", context)
        require(source, "cluster_r = [9, 9, 9, 9]", context)
        require(source, "cluster_dim1 = [40, 40, 60, 60]", context)
        require(source, "cluster_dim2 = [40, 60, 40, 60]", context)
        require(source, "cluster_dim3 = [60, 60, 40, 40]", context)
        require(source, "cluster_dim4 = [60, 40, 60, 40]", context)
        require(source, "2 * math.pi * random.random()", context)
        require(source, "cluster_r[i] * math.sqrt(random.random())", context)
        require(source, "numpy.float32", context)
        require(source, "random.random()", context)
        if not source.index(output_open) < source.index(header_write) < source.index(
            "for index in range(sum_np):", source.index(output_open)
        ):
            raise AssertionError(f"{context} header is not written before data rows")
        if "random.seed" in source:
            raise AssertionError(f"{context} unexpectedly adds a seed")

    stress_path = root / "Datasets/stress/generate_stress_test.py"
    stress = stress_path.read_text(encoding="utf-8")
    for fragment in (
        "N = 10**7",
        "D = 32",
        "K = 32",
        "center_box = (0, 100)",
        "cluster_std = 9.0",
        "points_per_cluster = N // K",
        "remainder = N % K",
        "centers = np.random.uniform(center_box[0], center_box[1], (K, D))",
        "np.random.randn(n_samples, D).astype(np.float32) * (cluster_std / 3.0)",
        "dtype=np.float32",
        "dtype=np.int32",
        'feature_header = ",".join([f"dim{i+1}" for i in range(D)])',
        'np.savetxt("stress_test_n1e7_d32_k32.csv", X, fmt=\'%.4f\', delimiter=\',\'',
        'np.savetxt("stress_test_labels.txt", labels, fmt=\'%d\'',
        "header=feature_header, comments=''",
    ):
        require(stress, fragment, stress_path.relative_to(root).as_posix())
    if "np.random.seed" in stress or "np.random.default_rng" in stress:
        raise AssertionError("stress generator unexpectedly adds a seed")

    local_path = root / "Datasets/local_memory_boundary/generate_local_memory_boundary.py"
    local = local_path.read_text(encoding="utf-8")
    for fragment in (
        "N = 40960",
        "D = 393",
        "DATA_SEED = 550",
        "PROTOTYPE_COUNT = 512",
        "BASE_PROTOTYPE_COUNT = 256",
        "REPLICAS_PER_PROTOTYPE = 80",
        "GAMMA = 0x9E3779B97F4A7C15",
        "MIX1 = 0xBF58476D1CE4E5B9",
        "MIX2 = 0x94D049BB133111EB",
        "DENOMINATOR = 2048",
        "DECIMAL_SCALE = 10**11",
        "base_mode = mode % BASE_PROTOTYPE_COUNT",
        "sign = 1 if mode < BASE_PROTOTYPE_COUNT else -1",
        "return 2 * ((splitmix64(key) >> 58) & 0x3F) - 63",
        "return 2 * ((splitmix64(key) >> 60) & 0xF) - 15",
        'OUTPUT_NAME = "local_memory_boundary_n40960_d393.csv"',
        "prototype_odd(base_mode, dimension, data_seed) * 256",
        "+ noise_odd(base_mode, replica, dimension, data_seed)",
        'csv_handle.write(",".join(f"f{index}" for index in range(D)) + "\\n")',
        'csv_handle.write(",".join(decimal_text(value) for value in row) + "\\n")',
    ):
        require(local, fragment, local_path.relative_to(root).as_posix())

    guide_context = guide_path.relative_to(root).as_posix()
    for heading in (
        "# Synthetic Datasets",
        "## Overview",
        "## Scalability datasets",
        "## Stress dataset",
        "## Local-memory-boundary dataset",
    ):
        require(guide, heading, guide_context)

    scalability_guide = normalized_guide.split(
        "## Scalability datasets", 1
    )[1].split("## Stress dataset", 1)[0]
    stress_guide = normalized_guide.split("## Stress dataset", 1)[1].split(
        "## Local-memory-boundary dataset", 1
    )[0]
    local_guide = normalized_guide.split(
        "## Local-memory-boundary dataset", 1
    )[1]

    for fragment in (
        "R = 9",
        "C0 = (40, 40, 60, 60)",
        "C1 = (40, 60, 60, 40)",
        "C2 = (60, 40, 40, 60)",
        "C3 = (60, 60, 40, 40)",
        "do not set an explicit random seed",
        "python3 generate_scalability_n1e5.py",
    ):
        require(scalability_guide, fragment, "scalability guide")
    for fragment in (
        "N = 10,000,000",
        "D = 32",
        "K = 32",
        "points per cluster = 312,500",
        "U[0,100)",
        "Normal(0, 3.0^2)",
        "standard deviation of `3.0`",
        "does not set an explicit random seed",
        "python3 generate_stress_test.py",
    ):
        require(stress_guide, fragment, "stress guide")
    for fragment in (
        "deterministic",
        "DATA_SEED = 550",
        "N = 40,960",
        "D = 393",
        "PROTOTYPE_COUNT = 512",
        "BASE_PROTOTYPE_COUNT = 256",
        "REPLICAS_PER_PROTOTYPE = 80",
        "python3 generate_local_memory_boundary.py --output-dir <directory>",
    ):
        require(local_guide, fragment, "local-memory guide")
    if "## Single-device-capacity" in guide:
        raise AssertionError("capacity workload returned to the dataset guide")

    dataset_readme = (root / "Datasets/README.md").read_text(encoding="utf-8")
    scalability_readme = (
        root / "Datasets/scalability/README.md"
    ).read_text(encoding="utf-8")
    require(
        dataset_readme,
        "[`SYNTHETIC_DATASETS.md`](SYNTHETIC_DATASETS.md)",
        "Datasets/README.md",
    )
    require(
        scalability_readme,
        "[`../SYNTHETIC_DATASETS.md`](../SYNTHETIC_DATASETS.md)",
        "Datasets/scalability/README.md",
    )

    print("dataset generator contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
