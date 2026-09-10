#!/usr/bin/env python3
"""Fail closed on CP-M dispatch or packet-scope regressions."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config-header", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    header = args.config_header.read_text(encoding="utf-8")
    forbidden = {
        "m_is_2": "runtime m=2 branch remains",
        "(m == 2)": "runtime m=2 comparison remains",
        "native::powr": "forbidden native powr appears",
        "qualified-enabled": "qualification decision appears in source",
        "production-enabled": "production enablement appears in source",
    }
    combined = source + "\n" + header
    for token, message in forbidden.items():
        if token in combined:
            raise SystemExit(f"{message}: {token}")
    if source.count("membership_power(ratio)") != 1:
        raise SystemExit("Path A/B/C do not share exactly one evaluator contract")
    if source.count("center_power(u)") != 1:
        raise SystemExit("center kernel does not use the host-selected evaluator")
    if "auto execute = [&](auto center_power, auto membership_power)" not in source:
        raise SystemExit("single outer host dispatch is missing")
    if "for (int iter = 0; iter < loop_limit; ++iter)" not in source:
        raise SystemExit("execution-mode loop is missing")
    loop = source.split("for (int iter = 0; iter < loop_limit; ++iter)", 1)[1]
    loop = loop.split("q.wait();\n  auto end", 1)[0]
    for token in ("special_value", "effective_path", "requested_policy"):
        if token in loop:
            raise SystemExit(f"m/policy dispatch leaked into fixed-20 loop: {token}")
    required = (
        "MPowerPolicy",
        "FuzzifierConfig",
        "CenterM200SquareEvaluator",
        "CenterGenericPowEvaluator",
        "MembershipM200IdentityEvaluator",
        "MembershipGenericPowEvaluator",
    )
    for token in required:
        if token not in header:
            raise SystemExit(f"missing public m-power core: {token}")
    production = header.split("struct FuzzifierConfig", 1)[1].split("};", 1)[0]
    for token in ("Stage" + "MPowerPolicy", "qualification", "candidate", "Telemetry"):
        if token in production:
            raise SystemExit(
                f"production fuzzifier config depends on private state: {token}"
            )
    resolver = header.split("FuzzifierConfig resolve_fuzzifier_config", 1)[1]
    resolver = resolver.split(");", 1)[0]
    if "Stage" + "MPowerPolicy" in resolver or "qualification" in resolver:
        raise SystemExit("production resolver exposes qualification controls")
    for token in (
        "Stage" + "MPowerPolicy",
        "MPower" + "QualificationConfig",
        "MPower" + "Telemetry",
        "Special" + "FuzzifierValue",
        "Center" + "M150MulSqrtEvaluator",
        "Center" + "M250SquareSqrtEvaluator",
        "Center" + "M300CubeEvaluator",
        "Membership" + "M150SquareEvaluator",
        "Membership" + "M250CbrtSquareEvaluator",
        "Membership" + "M300SqrtEvaluator",
    ):
        if token in combined:
            raise SystemExit(f"qualification-only m-power symbol remains: {token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
