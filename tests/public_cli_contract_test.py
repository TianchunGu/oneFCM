#!/usr/bin/env python3
"""Guard the retained public CLI vocabulary, defaults, and compatibility alias."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


EXPECTED_PUBLIC_OPTIONS = {
    "-i", "-o", "-c", "-e", "-n", "-d",
    "--iteration-protocol", "--max-iter", "--seed",
    "--fuzzifier", "--m-power-policy", "--force-generic-m",
    "--work-group-size", "--center-wg-size",
    "--membership-wg-size", "--change-wg-size",
}

COMPATIBILITY_NOTICE = (
    "Compatibility alias: --force-generic-m is equivalent to "
    "--m-power-policy generic."
)


def option_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])(--?[a-z][a-z0-9-]*)", text))


def base_command(binary: Path, dataset: Path, output: Path) -> list[str]:
    return [
        str(binary), "-i", str(dataset), "-o", str(output),
        "-c", "2", "-e", "0.001", "-n", "4", "-d", "2",
    ]


def run_case(binary: Path, dataset: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="public-cli-contract-") as tmp:
        output = Path(tmp) / "labels.csv"
        completed = subprocess.run(
            base_command(binary, dataset, output) + extra,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"retained public CLI case failed: {extra}\n"
                f"stdout={completed.stdout}\nstderr={completed.stderr}"
            )
        if not output.is_file():
            raise AssertionError(f"retained public CLI case produced no labels: {extra}")
        return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--execution-header", type=Path, required=True)
    parser.add_argument("--m-power-header", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    execution_header = args.execution_header.read_text(encoding="utf-8")
    m_power_header = args.m_power_header.read_text(encoding="utf-8")

    usage_block = source.split("if (argc < 13)", 1)[1].split("return 1;", 1)[0]
    allowed_block = source.split(
        "std::set<std::string> allowed_options", 1
    )[1].split("};", 1)[0]
    advertised = option_tokens(usage_block)
    accepted = option_tokens(allowed_block)
    if 'arg == "--force-generic-m"' not in source:
        raise AssertionError("compatibility alias is not accepted by the parser")
    accepted.add("--force-generic-m")

    if advertised != accepted:
        raise AssertionError(
            f"parser/usage mismatch: advertised={sorted(advertised)}, "
            f"accepted={sorted(accepted)}"
        )
    if accepted != EXPECTED_PUBLIC_OPTIONS:
        raise AssertionError(
            f"public CLI vocabulary changed: {sorted(accepted)}"
        )

    required_defaults = {
        "fixed20 protocol": "IterationProtocol protocol = IterationProtocol::Fixed20;",
        "max_iter=20": "int max_iter = 20;",
        "seed=550": "int seed = 550;",
        "fuzzifier text=2.0": 'std::string requested_m_text = "2.0";',
        "fuzzifier value=2.0": "double requested_m = 2.0;",
        "m-power policy=auto": "MPowerPolicy requested_policy = MPowerPolicy::Auto;",
    }
    for name, declaration in required_defaults.items():
        header = execution_header if name in {
            "fixed20 protocol", "max_iter=20", "seed=550"
        } else m_power_header
        if declaration not in header:
            raise AssertionError(f"public default changed: {name}")

    usage = subprocess.run(
        [str(args.binary)], text=True, capture_output=True, check=False
    )
    usage_text = usage.stdout + usage.stderr
    if usage.returncode == 0:
        raise AssertionError("empty invocation unexpectedly succeeded")
    if COMPATIBILITY_NOTICE not in usage_text:
        raise AssertionError("usage does not mark --force-generic-m as compatibility")
    if option_tokens(usage_text) != EXPECTED_PUBLIC_OPTIONS:
        raise AssertionError("runtime usage vocabulary differs from the source contract")

    retained = [
        "--iteration-protocol", "fixed20", "--max-iter", "20",
        "--seed", "550", "--fuzzifier", "2",
        "--m-power-policy", "auto", "--work-group-size", "2",
        "--center-wg-size", "2", "--membership-wg-size", "2",
        "--change-wg-size", "2",
    ]
    completed = run_case(args.binary, args.dataset, retained)
    for marker in (
        "Iteration protocol: fixed20",
        "Actual iterations: 20",
        "Stop reason: fixed20_complete",
    ):
        if marker not in completed.stdout:
            raise AssertionError(f"default contract marker missing: {marker}")
    run_case(args.binary, args.dataset, ["--fuzzifier", "2", "--force-generic-m"])
    print("public CLI contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
