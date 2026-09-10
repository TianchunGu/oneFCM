#!/usr/bin/env python3
"""Static gate for public WGS ownership and Scheme A/B structure."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-group-header", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    work_group_header = args.work_group_header.read_text(encoding="utf-8")
    for forbidden in (
        "select_work_group_size(",
        "clamp_work_group_size(",
        "desired_gs",
        "auto_gs",
        "requested_group_hint",
        "selected_group_size",
    ):
        if forbidden in source or forbidden in work_group_header:
            raise AssertionError(f"forbidden preselection/legacy field: {forbidden}")
    for required in (
        "cmeans::WorkGroupRequest work_group_request;",
        "work_group_request.center_membership_raw_requested",
        "work_group_request.change_raw_requested",
        "selected_tile_k * selected_tile_d * sizeof(float)",
        "select_work_group_or_throw",
    ):
        if required not in source:
            raise AssertionError(f"missing raw-request/selection contract: {required}")
    scheme_b_contract = (
        "if (K <= 32)",
        "constexpr size_t kSchemeBPrivateDistanceBound = 32;",
        "constexpr size_t kSchemeBLocalMemReserveBytes = 1024;",
        "constexpr size_t kSchemeBTargetResidency = 2;",
        "constexpr size_t kSchemeBTileKCandidates[] = {32, 16, 8, 4, 2, 1};",
        "selected_tile_k * selected_tile_d * sizeof(float)",
        "range<1>(selected_tile_k * selected_tile_d)",
        "membership_next_dev[cluster * N + j]",
    )
    for required in scheme_b_contract:
        if required not in source:
            raise AssertionError(f"missing Scheme A/B structural contract: {required}")
    if source.count("cluster_start += selected_tile_k") < 2:
        raise AssertionError("Scheme B no longer performs two K-tiled passes")
    if source.count("feature_start += selected_tile_d") < 2:
        raise AssertionError("Scheme B no longer performs two D-tiled passes")
    for forbidden in (
        "scheme_b_distance_scratch",
        "membership_distance_scratch",
    ):
        if forbidden in source:
            raise AssertionError(f"Scheme B persistent scratch remains: {forbidden}")
    for forbidden in (
        "WorkGroup" + "Telemetry",
        "observe_work_group_selection",
        "record_work_group_submission_failure",
        "maybe_inject_work_group_failure",
    ):
        if forbidden in source or forbidden in work_group_header:
            raise AssertionError(f"internal WGS diagnostic seam remains: {forbidden}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
