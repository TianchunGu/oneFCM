#!/usr/bin/env python3
"""Guard the production build, dependency, CLI, and output boundary."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


FORBIDDEN_OPTIONS = {
    "--diagnostics-level": "joint",
    "--diagnostics-out": "diagnostics.json",
    "--manifest-out": "manifest.json",
    "--center-m-power-policy": "candidate",
    "--membership-m-power-policy": "candidate",
}


def base_command(binary: Path, dataset: Path, output: Path) -> list[str]:
    return [
        str(binary), "-i", str(dataset), "-o", str(output),
        "-c", "2", "-e", "0.2", "-n", "4", "-d", "2",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root-cmake", type=Path, required=True)
    parser.add_argument("--cmeans-cmake", type=Path, required=True)
    parser.add_argument("--public-header", type=Path, required=True)
    args = parser.parse_args()

    usage = subprocess.run(
        [str(args.binary)], text=True, capture_output=True, check=False
    )
    usage_text = usage.stdout + usage.stderr
    for option in FORBIDDEN_OPTIONS:
        if option in usage_text:
            raise AssertionError(f"production usage advertises {option}")

    root_cmake = args.root_cmake.read_text(encoding="utf-8")
    if "cmake_minimum_required(VERSION 3.16)" not in root_cmake:
        raise AssertionError("root CMake minimum is not 3.16")
    for token in (
        "FCM_ENABLE_CP_DIAG_" + "TESTS",
        "fcm_build_manifest",
        "FCM_REQUIRE_GIT_" + "IDENTITY",
        "FCM_ENABLE_T" + "P6_QUALIFICATION_CONTROLS",
        "FCM_ENABLE_T" + "P7_CAPACITY_PROBE",
        "FCM_ENABLE_MEMBERSHIP_SCHEME_B_" + "QUALIFICATION",
        "FCM_ENABLE_M_SPECIALIZATION_" + "TEST_CONTROLS",
    ):
        if token in root_cmake:
            raise AssertionError(f"internal release seam remains: {token}")
    cmeans_cmake = args.cmeans_cmake.read_text(encoding="utf-8")
    if "cmake_minimum_required(VERSION 3.16)" not in cmeans_cmake:
        raise AssertionError("cmeans CMake minimum is not 3.16")
    link_block = cmeans_cmake.split(
        "target_link_libraries(${TARGET_NAME} PRIVATE", 1
    )[1].split(")", 1)[0]
    for target in ("fcm_diagnostics", "fcm_build_manifest"):
        if target in link_block:
            raise AssertionError(f"production link block contains {target}")
    public_header = args.public_header.read_text(encoding="utf-8")
    active_build_contract = root_cmake + cmeans_cmake + public_header
    for token in (
        "/Users/",
        "/home/",
        "/root/",
        "C:\\Users",
        "sycl_workspace",
        "libmkl_rt",
        "#include <mkl.h>",
    ):
        if token in active_build_contract:
            raise AssertionError(f"non-portable build dependency remains: {token}")
    for legacy_link in ("\n    pthread\n", "\n    m\n", "\n    dl\n"):
        if legacy_link in cmeans_cmake:
            raise AssertionError(
                f"unused direct system link remains: {legacy_link.strip()}"
            )
    for duplicate in (
        "target_compile_options(${TARGET_NAME} PRIVATE -fsycl",
        "target_link_options(${TARGET_NAME} PRIVATE -fsycl",
        "target_compile_definitions(${TARGET_NAME} PRIVATE _GPU_",
        "target_compile_definitions(${TARGET_NAME} PRIVATE _CPU_",
    ):
        if duplicate in cmeans_cmake:
            raise AssertionError(f"duplicate SYCL target contract remains: {duplicate}")
    for required in (
        "fcm_sycl_target_contract",
        "fcm_onedpl",
        "FCM_ONEDPL_ROOT",
        "FCM_AMD_SYCL_BUILD_ROOT",
        "FCM_ROCM_ROOT",
    ):
        if required not in active_build_contract:
            raise AssertionError(f"release build contract missing: {required}")
    public_signature = public_header.split("int fuzzy_c_means_standard(", 1)[1]
    public_signature = public_signature.split(");", 1)[0]
    for type_name in (
        "Diagnostics" + "Level",
        "Fcm" + "Diagnostics",
        "WorkGroup" + "Telemetry",
    ):
        if type_name in public_signature:
            raise AssertionError(f"production API exposes {type_name}")

    with tempfile.TemporaryDirectory(prefix="release-production-boundary-") as tmp:
        root = Path(tmp)
        for option, value in FORBIDDEN_OPTIONS.items():
            case_root = root / option.removeprefix("--")
            case_root.mkdir()
            completed = subprocess.run(
                base_command(args.binary, args.dataset, case_root / "labels.csv")
                + [option, value],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                raise AssertionError(f"production CLI accepted {option}")
            if (case_root / "labels.csv").exists():
                raise AssertionError(f"{option} reached algorithm execution")

        ordinary = root / "ordinary"
        ordinary.mkdir()
        completed = subprocess.run(
            base_command(args.binary, args.dataset, ordinary / "labels.csv")
            + ["--work-group-size", "2"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"production explicit-WGS run failed\n{completed.stderr}"
            )
        if sorted(path.name for path in ordinary.iterdir()) != [
            "centers.csv", "labels.csv", "time.csv"
        ]:
            raise AssertionError("production run generated non-ordinary files")
        for marker in (
            "Iteration protocol: fixed20",
            "Actual iterations: 20",
            "Stop reason: fixed20_complete",
        ):
            if marker not in completed.stdout:
                raise AssertionError(f"missing production stdout marker: {marker}")

        input_root = root / "input-contract"
        input_root.mkdir()
        valid_cases = {
            "normal": (
                "x,y\n1,10\n2,20\n3,30\n4,40\n",
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            ),
            "extra-columns": (
                "x,y,ignored\n1,10,100\n2,20,200\n3,30,300\n4,40,400\n",
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            ),
            "extra-rows": (
                "x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n",
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            ),
        }
        for name, (content, expected_values) in valid_cases.items():
            case_root = input_root / name
            case_root.mkdir()
            dataset = case_root / "input.csv"
            dataset.write_text(content, encoding="ascii")
            output = case_root / "labels.csv"
            completed = subprocess.run(
                base_command(args.binary, dataset, output),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"valid input case failed: {name}\n"
                    f"stdout={completed.stdout}\nstderr={completed.stderr}"
                )
            rows = [
                [float(value) for value in line.split(",")[:2]]
                for line in output.read_text(encoding="ascii").splitlines()
            ]
            if rows != expected_values:
                raise AssertionError(
                    f"valid input layout changed: {name}: {rows}"
                )

        invalid_cases = {
            "missing-header": ("", "input CSV is missing a header row"),
            "header-too-narrow": (
                "x\n1,10\n2,20\n3,30\n4,40\n",
                "header has fewer than 2 columns",
            ),
            "short-row": (
                "x,y\n1,10\n2\n3,30\n4,40\n",
                "row 2 has fewer than 2 columns",
            ),
            "malformed-numeric": (
                "x,y\n1,10\nbad,20\n3,30\n4,40\n",
                "invalid numeric value at row 2, column 1",
            ),
            "nan": (
                "x,y\n1,10\nnan,20\n3,30\n4,40\n",
                "non-finite numeric value at row 2, column 1",
            ),
            "positive-inf": (
                "x,y\n1,10\ninf,20\n3,30\n4,40\n",
                "non-finite numeric value at row 2, column 1",
            ),
            "negative-inf": (
                "x,y\n1,10\n-inf,20\n3,30\n4,40\n",
                "non-finite numeric value at row 2, column 1",
            ),
            "fewer-rows": (
                "x,y\n1,10\n2,20\n3,30\n",
                "expected 4 data rows but only found 3",
            ),
        }
        for name, (content, error_fragment) in invalid_cases.items():
            case_root = input_root / name
            case_root.mkdir()
            dataset = case_root / "input.csv"
            dataset.write_text(content, encoding="ascii")
            output = case_root / "labels.csv"
            completed = subprocess.run(
                base_command(args.binary, dataset, output),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                raise AssertionError(f"invalid input case succeeded: {name}")
            if error_fragment not in completed.stderr:
                raise AssertionError(
                    f"invalid input diagnostic mismatch: {name}\n"
                    f"stdout={completed.stdout}\nstderr={completed.stderr}"
                )
            if output.exists():
                raise AssertionError(f"invalid input produced labels: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
