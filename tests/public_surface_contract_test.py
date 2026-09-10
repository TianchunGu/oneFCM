#!/usr/bin/env python3

import argparse
import pathlib
import re
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-header", required=True)
    parser.add_argument("--work-group-header", required=True)
    parser.add_argument("--work-group-source", required=True)
    parser.add_argument("--production-source", required=True)
    args = parser.parse_args()

    public_header = pathlib.Path(args.public_header).read_text()
    work_group_header = pathlib.Path(args.work_group_header).read_text()
    work_group_source = pathlib.Path(args.work_group_source).read_text()
    production_source = pathlib.Path(args.production_source).read_text()

    errors: list[str] = []
    for namespace in ("std", "sycl"):
        if re.search(rf"\busing\s+namespace\s+{namespace}\s*;", public_header):
            errors.append(f"public header imports namespace {namespace}")

    forbidden_public_includes = {
        "cmath", "cstdlib", "fstream", "functional", "iostream", "sstream",
        "string", "vector",
    }
    includes = set(re.findall(r"^#include\s+<([^>]+)>", public_header, re.M))
    unexpected = sorted(includes & forbidden_public_includes)
    if unexpected:
        errors.append("unused public includes remain: " + ", ".join(unexpected))

    legacy_declarations = (
        r"\bsycl::event\s+parallel_initialize_membership_matrix\s*\(",
        r"\bvoid\s+update_cluster_centers\s*\(",
        r"\bvoid\s+update_membership_values\s*\(",
        r"\bfloat\s+change_membership_matrix\s*\(",
        r"\bvoid\s+classify\s*\(",
    )
    for declaration in legacy_declarations:
        if re.search(declaration, public_header):
            errors.append(f"stale public declaration remains: {declaration}")

    if "fuzzy_c_means_standard" not in public_header:
        errors.append("required production entry point is missing")

    dead_work_group_surface = (
        "SubmissionFailed", "cache_tile_k", "failure_stage",
        "exception_category", "failure_message",
    )
    combined_work_group = work_group_header + work_group_source
    for symbol in dead_work_group_surface:
        if symbol in combined_work_group:
            errors.append(f"dead work-group surface remains: {symbol}")

    if "cache_tile_k" in production_source:
        errors.append("production source still writes removed cache_tile_k metadata")

    if errors:
        print("public surface contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("public surface contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
