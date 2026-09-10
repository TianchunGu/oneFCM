#!/usr/bin/env python3
"""Tiny public fuzzifier regression using production outputs only."""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path


EXPECTED_TIME_COLUMNS = [
    "read_data_time_ms",
    "write_data_time_ms",
    "fcm_time_ms_total",
    "fcm_time_ms_avg_per_iter",
    "iter_wall_time_ms_avg",
    "first_iter_wall_time_ms",
    "other_iter_wall_time_ms_avg",
    "init_membership_kernel_time_ms",
    "classify_kernel_time_ms",
    "update_membership_kernel_time_ms_avg",
    "update_cluster_centers_kernel_time_ms_avg",
    "change_matrix_kernel_time_ms_avg",
    "iteration_count",
    "change_calls",
]


def numeric_values(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for field in line.split(","):
            if field.strip():
                values.append(float(field))
    if not values or not all(math.isfinite(value) for value in values):
        raise AssertionError(f"{path.name} is empty or nonfinite")
    return values


def run_case(binary: Path, dataset: Path, name: str, extra: list[str]) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"m-power-public-{name}-") as tmp:
        root = Path(tmp)
        labels = root / "labels.csv"
        command = [
            str(binary), "-i", str(dataset), "-o", str(labels),
            "-c", "2", "-e", "0.001", "-n", "4", "-d", "2",
        ] + extra
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            raise AssertionError(
                f"{name} failed\nstdout={completed.stdout}\nstderr={completed.stderr}"
            )
        centers = root / "centers.csv"
        timing = root / "time.csv"
        for output in (labels, centers, timing):
            if not output.is_file():
                raise AssertionError(f"{name} did not produce {output.name}")
        if (root / "diagnostics.json").exists() or (root / "manifest.json").exists():
            raise AssertionError(f"{name} generated an internal artifact")
        if "Iteration protocol: fixed20" not in completed.stdout:
            raise AssertionError(f"{name} protocol output differs")
        if "Actual iterations: 20" not in completed.stdout:
            raise AssertionError(f"{name} iteration output differs")
        if "Stop reason: fixed20_complete" not in completed.stdout:
            raise AssertionError(f"{name} stop output differs")
        header = timing.read_text(encoding="utf-8").splitlines()[0].split(",")
        if header != EXPECTED_TIME_COLUMNS:
            raise AssertionError(f"{name} timing schema differs")
        return {
            "labels": labels.read_bytes(),
            "centers": numeric_values(centers),
        }


def reject_case(binary: Path, dataset: Path, name: str, extra: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix=f"m-power-invalid-{name}-") as tmp:
        root = Path(tmp)
        labels = root / "labels.csv"
        command = [
            str(binary), "-i", str(dataset), "-o", str(labels),
            "-c", "2", "-e", "0.001", "-n", "4", "-d", "2",
        ] + extra
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode == 0 or labels.exists():
            raise AssertionError(f"invalid {name} reached algorithm execution")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        "m150": ["--fuzzifier", "1.5"],
        "m200_auto": ["--fuzzifier", "2", "--m-power-policy", "auto"],
        "m200_generic": ["--fuzzifier", "2", "--m-power-policy", "generic"],
        "m200_forced_generic": ["--fuzzifier", "2", "--force-generic-m"],
        "m250": ["--fuzzifier", "2.5"],
        "m300": ["--fuzzifier", "3"],
        "m170": ["--fuzzifier", "1.7"],
        "m425": ["--fuzzifier", "4.25"],
        "m20001": ["--fuzzifier", "2.0001"],
        "m19999": ["--fuzzifier", "1.9999"],
    }
    results = {
        name: run_case(args.binary, args.dataset, name, extra)
        for name, extra in cases.items()
    }

    specialized = results["m200_auto"]
    for name in ("m200_generic", "m200_forced_generic"):
        generic = results[name]
        if specialized["labels"] != generic["labels"]:
            raise AssertionError(f"m=2 specialized/{name} labels differ")
        if len(specialized["centers"]) != len(generic["centers"]):
            raise AssertionError(f"m=2 specialized/{name} center shape differs")
        if not all(
            math.isclose(left, right, rel_tol=1.0e-5, abs_tol=1.0e-6)
            for left, right in zip(specialized["centers"], generic["centers"])
        ):
            raise AssertionError(f"m=2 specialized/{name} centers differ")

    invalid_cases = {
        "m_le_1": ["--fuzzifier", "1"],
        "nan": ["--fuzzifier", "NaN"],
        "positive_inf": ["--fuzzifier", "+Inf"],
        "negative_inf": ["--fuzzifier", "-Inf"],
        "effective_float_overflow": ["--fuzzifier", "1e39"],
        "effective_float_rounds_to_one": ["--fuzzifier", "1.00000001"],
        "malformed": ["--fuzzifier", "2.0abc"],
        "missing": ["--fuzzifier"],
        "duplicate": ["--fuzzifier", "2", "--fuzzifier", "2.5"],
        "policy_alias_conflict": [
            "--m-power-policy", "auto", "--force-generic-m"
        ],
    }
    for name, extra in invalid_cases.items():
        reject_case(args.binary, args.dataset, name, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
