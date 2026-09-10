#!/usr/bin/env python3
"""Guard the public README against CLI drift and release-scope overclaim."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EXPECTED_PUBLIC_OPTIONS = {
    "-i", "-o", "-c", "-e", "-n", "-d",
    "--iteration-protocol", "--max-iter", "--seed",
    "--fuzzifier", "--m-power-policy", "--force-generic-m",
    "--work-group-size", "--center-wg-size",
    "--membership-wg-size", "--change-wg-size",
}

EXPECTED_TIME_FIELDS = {
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
}

EXPECTED_DYNAMIC_THRESHOLDS = {
    "2.91168708e-05",
    "2.54213828e-05",
    "0.00288921618",
    "0.7965019822354785",
    "9.3340881e-05",
    "0.000238835826",
    "0.000116765506",
    "0.00046005848",
    "1.4935728761491072",
    "0.000746667443",
    "1.04382643e-05",
    "1.04755172e-05",
    "7.71135183e-06",
    "0.10741036858604799",
    "1.06915841e-05",
}

REQUIRED_SECTIONS = {
    "Overview": "## Overview",
    "Features": "## Features",
    "Requirements": "## Requirements",
    "Build": "## Build",
    "Run": "## Run",
    "Input/output": "## Input and output",
    "Modes": "## Execution modes",
    "Seed/fuzzifier": "## Seed and fuzzifier",
    "WGS": "## Work-group sizes",
    "Testing": "## Testing",
    "Manuscript datasets": "## Datasets used in the manuscript",
    "Reproducibility scope": "## Reproducibility scope",
    "Citation": "## Citation",
    "License": "## License",
}

FORBIDDEN_OPTIONS = {
    "--diagnostics-level",
    "--diagnostics-out",
    "--manifest-out",
    "--center-m-power-policy",
    "--membership-m-power-policy",
}


def option_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])(--?[a-z][a-z0-9-]*)", text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public-header", type=Path, required=True)
    parser.add_argument("--m-power-source", type=Path, required=True)
    args = parser.parse_args()

    readme = args.readme.read_text(encoding="utf-8")
    source = args.source.read_text(encoding="utf-8")
    public_header = args.public_header.read_text(encoding="utf-8")
    m_power_source = args.m_power_source.read_text(encoding="utf-8")
    dataset_readme_path = (
        args.readme.parent / "Datasets/scalability/README.md"
    )
    if not dataset_readme_path.is_file():
        raise AssertionError("scalability dataset README missing")
    dataset_readme = dataset_readme_path.read_text(encoding="utf-8")
    normalized_dataset_readme = " ".join(dataset_readme.split())
    dataset_overview_path = args.readme.parent / "Datasets/README.md"
    if not dataset_overview_path.is_file():
        raise AssertionError("dataset overview README missing")
    dataset_overview = dataset_overview_path.read_text(encoding="utf-8")
    citation_path = args.readme.parent / "CITATION.cff"
    if not citation_path.is_file():
        raise AssertionError("machine-readable citation metadata missing")
    citation = citation_path.read_text(encoding="utf-8")
    license_path = args.readme.parent / "LICENSE"
    if not license_path.is_file():
        raise AssertionError("README license target missing: LICENSE")
    license_text = license_path.read_text(encoding="utf-8")

    for name, heading in REQUIRED_SECTIONS.items():
        if heading not in readme:
            raise AssertionError(f"README section missing: {name}")

    for heading in (
        "## Hardware verification boundary",
        "## Reproducibility",
    ):
        if re.search(rf"^{re.escape(heading)}$", readme, flags=re.MULTILINE):
            raise AssertionError(f"removed README section returned: {heading}")
    if "````" in readme:
        raise AssertionError("README contains a four-backtick Markdown fence")

    missing_time_fields = sorted(EXPECTED_TIME_FIELDS - set(readme.split("`")))
    if missing_time_fields:
        raise AssertionError(f"README time.csv fields missing: {missing_time_fields}")
    missing_thresholds = sorted(
        value for value in EXPECTED_DYNAMIC_THRESHOLDS if value not in readme
    )
    if missing_thresholds:
        raise AssertionError(
            f"README Dynamic thresholds missing: {missing_thresholds}"
        )
    for workload, seed in (
        ("GID", "700"),
        ("KDD Cup 1999", "700"),
        ("Electricity Load", "701"),
    ):
        if not re.search(rf"\|\s*{re.escape(workload)}\s*\|\s*{seed}\s*\|", readme):
            raise AssertionError(
                f"README Dynamic workload/seed row missing: {workload}/{seed}"
            )

    for license_contract in (
        "Copyright (c) 2026, Tianchun Gu and Guanlin He",
        "Redistribution and use in source and binary forms",
        "Neither the name of the copyright holder",
    ):
        if license_contract not in license_text:
            raise AssertionError(
                f"LICENSE BSD-3-Clause contract missing: {license_contract}"
            )

    cli_section = readme.split("### Command-line interface", 1)[1]
    cli_section = cli_section.split("\n## ", 1)[0]
    documented = {
        match.group(1)
        for match in re.finditer(r"^\| `([^`]+)` \|", cli_section, re.MULTILINE)
    }
    usage_block = source.split("if (argc < 13)", 1)[1].split("return 1;", 1)[0]
    accepted_block = source.split(
        "std::set<std::string> allowed_options", 1
    )[1].split("};", 1)[0]
    accepted = option_tokens(accepted_block)
    if 'arg == "--force-generic-m"' not in source:
        raise AssertionError("parser no longer accepts the documented alias")
    accepted.add("--force-generic-m")
    advertised = option_tokens(usage_block)
    if documented != EXPECTED_PUBLIC_OPTIONS:
        raise AssertionError(
            f"README CLI vocabulary changed: {sorted(documented)}"
        )
    if documented != accepted or documented != advertised:
        raise AssertionError("README, parser, and runtime usage CLI differ")

    classify_block = source.split("void classify(", 1)[1].split(
        "template <typename CenterPowerEvaluator", 1
    )[0]
    if not re.search(r"\blabel\s*=\s*i\s*\+\s*1\s*;", classify_block):
        raise AssertionError("production classify no longer assigns one-based labels")
    if not re.search(r"\bl\s*\[\s*id\s*\]\s*=\s*label\s*;", classify_block):
        raise AssertionError("production classify no longer stores the selected label")
    if "1-based cluster label" not in readme or "0-based cluster label" in readme:
        raise AssertionError("README cluster-label indexing is not one-based")
    if (
        "Output array of one-based cluster labels in the range 1..K."
        not in public_header
    ):
        raise AssertionError("public header cluster-label contract is not one-based")

    for source_contract in (
        "const float effective = static_cast<float>(requested);",
        "if (!ieee_finite_float(effective) || effective <= 1.0f)",
        "const float effective_m = static_cast<float>(requested_m);",
        "if (!ieee_finite_float(effective_m) || effective_m <= 1.0f)",
        "config.membership_exponent_q = 1.0f / (effective_m - 1.0f);",
        "if (!ieee_finite_float(config.membership_exponent_q))",
    ):
        if source_contract not in m_power_source:
            raise AssertionError(
                f"production fuzzifier boundary changed: {source_contract}"
            )
    if "Any finite fuzzifier" in readme:
        raise AssertionError("README overstates the fuzzifier range")
    for documented_boundary in (
        "production single-precision representation",
        "finite membership exponent",
    ):
        if documented_boundary not in readme:
            raise AssertionError(
                f"README fuzzifier boundary missing: {documented_boundary}"
            )
    if "subproject requires 3.16" in readme:
        raise AssertionError("README overstates the CMake source contract")

    for option in FORBIDDEN_OPTIONS:
        if option in readme:
            raise AssertionError(f"removed option appears in README: {option}")

    for token in (
        "/Users/", "/home/", "C:\\Users", "elsarticle/", "revision/",
        "experiments/", "qualification", "diagnostic-only",
        "REPRODUCIBILITY" + ".md", "release-" + "provenance.json",
        "PROVENANCE_" + "GAP", "frozen " + "source",
        "historical " + "source",
        "not revalidated in this documentation stage",
        "No license file is currently included",
    ):
        if token in readme:
            raise AssertionError(f"non-public path or workflow term in README: {token}")

    for pattern in (
        r"state[- ]of[- ]the[- ]art", r"\bsuperior\b", r"\boutperform",
        r"\bspeedups?\b",
    ):
        if re.search(pattern, readme, flags=re.IGNORECASE):
            raise AssertionError(f"unsupported README claim matches: {pattern}")

    for required in (
        "classical fuzzy c-means",
        "not a new clustering algorithm",
        "The documented release workflow assumes CMake 3.16 or newer.",
        "`CUDA_GPU_ARCH` to match the target GPU",
        "not a universal NVIDIA setting",
        "`SYCL_LLVM_ROOT`",
        "`SYCL_BUILD_ROOT`",
        "`FCM_AMD_SYCL_COMPILER`",
        "`ROCM_PATH`",
        "`AMD_GPU_ARCH`",
        "does not represent a verified FPGA hardware deployment",
        "C++17 SYCL",
        "oneDPL",
        "The accepted GPU target tokens are `GPU`, `INTEL_ARC_B580`, and `AMD_GPU`",
        "first line to be a header",
        "Missing or narrow headers",
        "non-finite values fail with an error",
        "Extra columns and rows are allowed and ignored",
        "1-based cluster label",
        "fcm_time_ms_total / iteration_count",
        "not the iterative-loop time",
        "membership initialization occurs before this interval",
        "classification and final result copies occur after it",
        "When profiling is disabled, these fields are reported as `0`.",
        "exactly 20 iterations",
        "19 checks",
        "change < threshold",
        "first convergence check occurs after the second completed iteration",
        "does not define a universal default epsilon",
        "Dynamic thresholds used in the manuscript experiments",
        "`max_iter = 100`",
        "should therefore not be compared directly across implementations",
        "FCM_ENABLE_PROFILING=ON",
        "oneapi::dpl::minstd_rand",
        "`m = 1.5`, `m = 2.5`, and `m = 3`",
        "BSD 3-Clause License",
        "[LICENSE](LICENSE)",
        "Datasets/scalability/",
        "separately licensed third-party components",
        "CeCILL-C v1.0",
        "Large third-party datasets",
        "real-world preprocessing pipelines",
        "not packaged here",
        "[`CITATION.cff`](CITATION.cff)",
    ):
        if required not in readme:
            raise AssertionError(f"README contract statement missing: {required}")

    for required in (
        "without an explicit seed",
        "generated values may differ between runs",
        "LICENSE-CeCILL-C",
    ):
        if required not in normalized_dataset_readme:
            raise AssertionError(
                f"scalability dataset README contract missing: {required}"
            )

    for required in (
        "## Generator requirements",
        "Scalability generators: Python 3, NumPy, and Matplotlib.",
        "Stress generator: Python 3 and NumPy.",
        "Local-memory boundary generator: Python 3 standard library.",
        "## Generator output contract",
        "comma-separated feature CSV with one header row",
        "passed directly to oneFCM with the `-i` option",
    ):
        if required not in dataset_overview:
            raise AssertionError(
                f"dataset overview contract missing: {required}"
            )

    for required in (
        "cff-version: 1.2.0",
        "type: software",
        "title: \"oneFCM: An Efficient and Portable SYCL/oneAPI Implementation of Fuzzy c-Means Clustering\"",
        "family-names: Gu",
        "given-names: Tianchun",
        "https://orcid.org/0009-0008-1207-9840",
        "family-names: He",
        "given-names: Guanlin",
        "https://orcid.org/0000-0003-3753-3671",
        "version: 1.0.0",
        "license: BSD-3-Clause",
        'repository-code: "https://github.com/TianchunGu/oneFCM"',
    ):
        if required not in citation:
            raise AssertionError(f"CITATION.cff contract missing: {required}")
    for forbidden_key in (
        "doi",
        "url",
        "date-released",
    ):
        if re.search(rf"^{re.escape(forbidden_key)}\s*:", citation, re.MULTILINE):
            raise AssertionError(
                f"unverified CITATION.cff field present: {forbidden_key}"
            )

    for relative in (
        "LICENSE", "tests/README.md", "tests/fixtures/tiny.csv",
        "Datasets/scalability/README.md",
        "Datasets/scalability/LICENSE-CeCILL-C",
        "Datasets/README.md", "CITATION.cff",
    ):
        if not (args.readme.parent / relative).is_file():
            raise AssertionError(f"README link target missing: {relative}")

    print("README contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
