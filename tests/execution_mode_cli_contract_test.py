#!/usr/bin/env python3
"""Exercise the public execution-mode parser without scientific workloads."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def command(binary: Path, dataset: Path, output: Path) -> list[str]:
    return [
        str(binary),
        "-i", str(dataset),
        "-o", str(output),
        "-c", "2",
        "-e", "0.001",
        "-n", "4",
        "-d", "2",
    ]


def run_case(
    binary: Path,
    dataset: Path,
    extra: list[str],
    should_pass: bool,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="execution-mode-cli-") as tmp:
        output = Path(tmp) / "labels.csv"
        completed = subprocess.run(
            command(binary, dataset, output) + extra,
            text=True,
            capture_output=True,
            check=False,
        )
        if (completed.returncode == 0) != should_pass:
            raise SystemExit(
                f"unexpected result for {extra}: rc={completed.returncode}\n"
                f"stdout={completed.stdout}\nstderr={completed.stderr}"
            )
        if not should_pass and output.exists():
            raise SystemExit(f"invalid configuration invoked FCM: {extra}")
        if should_pass:
            expected_outputs = {"labels.csv", "centers.csv", "time.csv"}
            actual_outputs = {path.name for path in Path(tmp).iterdir()}
            if actual_outputs != expected_outputs:
                raise SystemExit(
                    f"unexpected output set for {extra}: {sorted(actual_outputs)}"
                )
        return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    run_case(args.binary, args.dataset, [], True)
    run_case(
        args.binary,
        args.dataset,
        ["--iteration-protocol", "fixed20"],
        True,
    )
    run_case(
        args.binary,
        args.dataset,
        ["--iteration-protocol", "fixed20", "--max-iter", "20"],
        True,
    )
    run_case(
        args.binary,
        args.dataset,
        ["--iteration-protocol", "fixed20", "--max-iter", "100"],
        False,
    )
    dynamic = run_case(
        args.binary,
        args.dataset,
        ["--iteration-protocol", "native-dynamic", "--max-iter", "100"],
        True,
    )
    if "Iteration protocol: native-dynamic" not in dynamic.stdout:
        raise SystemExit("Dynamic smoke did not report native-dynamic protocol")
    if "Actual iterations:" not in dynamic.stdout:
        raise SystemExit("Dynamic smoke did not report actual iterations")
    if not any(
        marker in dynamic.stdout
        for marker in (
            "Stop reason: native_converged",
            "Stop reason: max_iter_reached",
        )
    ):
        raise SystemExit("Dynamic smoke did not report a public stop reason")
    run_case(
        args.binary,
        args.dataset,
        ["--iteration-protocol", "native-dynamic"],
        False,
    )
    run_case(
        args.binary,
        args.dataset,
        ["--iteration-protocol", "native-dynamic", "--max-iter", "0"],
        False,
    )
    run_case(
        args.binary,
        args.dataset,
        ["--iteration-protocol", "dynamic", "--max-iter", "100"],
        False,
    )
    run_case(args.binary, args.dataset, ["--seed", "700"], True)
    run_case(
        args.binary,
        args.dataset,
        ["--seed", "700", "--seed", "701"],
        False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
