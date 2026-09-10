#!/usr/bin/env python3
"""Guard the exact public source tree and its publication-facing path hygiene."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path


EXPECTED_FILES = {
    ".dockerignore",
    ".gitignore",
    "CITATION.cff",
    "CMakeLists.txt",
    "Datasets/README.md",
    "Datasets/SYNTHETIC_DATASETS.md",
    "Datasets/local_memory_boundary/generate_local_memory_boundary.py",
    "Datasets/scalability/LICENSE-CeCILL-C",
    "Datasets/scalability/README.md",
    "Datasets/scalability/generate_scalability_n1e5.py",
    "Datasets/scalability/generate_scalability_n1e6.py",
    "Datasets/scalability/generate_scalability_n1e7.py",
    "Datasets/scalability/generate_scalability_n1e8.py",
    "Datasets/stress/generate_stress_test.py",
    "LICENSE",
    "README.md",
    "cmeans/CMakeLists.txt",
    "cmeans/include/cmeans/cmeans.h",
    "cmeans/include/cmeans/convergence.h",
    "cmeans/include/cmeans/execution_mode.h",
    "cmeans/include/cmeans/m_power_config.h",
    "cmeans/include/cmeans/work_group_config.h",
    "cmeans/src/cmeans.cpp",
    "cmeans/src/execution_mode.cpp",
    "cmeans/src/m_power_config.cpp",
    "cmeans/src/work_group_config.cpp",
    "containers/Dockerfile.platform-a",
    "containers/README.md",
    "tests/CMakeLists.txt",
    "tests/README.md",
    "tests/container_contract_test.py",
    "tests/dataset_generator_contract_test.py",
    "tests/dynamic_control_test.cpp",
    "tests/execution_mode_cli_contract_test.py",
    "tests/execution_mode_config_test.cpp",
    "tests/fixtures/tiny.csv",
    "tests/m_power_config_test.cpp",
    "tests/m_power_public_regression_test.py",
    "tests/m_power_source_contract_test.py",
    "tests/membership_init_timing_boundary_contract_test.py",
    "tests/normalization_layout_test.cpp",
    "tests/normalization_nonfinite_test.cpp",
    "tests/production_api_smoke_test.cpp",
    "tests/production_release_boundary_test.py",
    "tests/public_comment_contract_test.py",
    "tests/public_cli_contract_test.py",
    "tests/public_headers_compile_test.cpp",
    "tests/public_surface_contract_test.py",
    "tests/public_tree_contract_test.py",
    "tests/readme_contract_test.py",
    "tests/work_group_selection_test.cpp",
    "tests/work_group_source_contract_test.py",
    "util/CMakeLists.txt",
    "util/include/util/utils.h",
    "util/src/utils.cpp",
}

EXPECTED_GENERATOR_SHA256 = {
    "Datasets/scalability/generate_scalability_n1e5.py":
        "a5e90d9f14a5bdbffbababd69a9adbc41b523b23da693fb88376770f3acfb7d6",
    "Datasets/scalability/generate_scalability_n1e6.py":
        "835416707f1eab059962f7c0478d0b0ac0f46f88ea37f502131d368911fa64ae",
    "Datasets/scalability/generate_scalability_n1e7.py":
        "f2a77fb94abdaec503332bf354de943e94aab63f4bfcbf1f023be56af9b943ff",
    "Datasets/scalability/generate_scalability_n1e8.py":
        "dab6a8234367df62874d31d8fd02f01d7b11a42290a89b081382417fe521bbbb",
}

PUBLIC_TEXT_FILES = {
    ".dockerignore",
    ".gitignore",
    "CITATION.cff",
    "CMakeLists.txt",
    "Datasets/README.md",
    "Datasets/SYNTHETIC_DATASETS.md",
    "Datasets/scalability/README.md",
    "LICENSE",
    "README.md",
    "cmeans/CMakeLists.txt",
    "cmeans/include/cmeans/cmeans.h",
    "cmeans/include/cmeans/convergence.h",
    "cmeans/include/cmeans/execution_mode.h",
    "cmeans/include/cmeans/m_power_config.h",
    "cmeans/include/cmeans/work_group_config.h",
    "cmeans/src/cmeans.cpp",
    "cmeans/src/execution_mode.cpp",
    "cmeans/src/m_power_config.cpp",
    "cmeans/src/work_group_config.cpp",
    "containers/Dockerfile.platform-a",
    "containers/README.md",
    "tests/CMakeLists.txt",
    "tests/README.md",
    "util/CMakeLists.txt",
    "util/include/util/utils.h",
    "util/src/utils.cpp",
}

ACTIVE_PUBLIC_SCRIPTS = {
    "tests/container_contract_test.py",
    "tests/dataset_generator_contract_test.py",
    "tests/execution_mode_cli_contract_test.py",
    "tests/m_power_public_regression_test.py",
    "tests/m_power_source_contract_test.py",
    "tests/membership_init_timing_boundary_contract_test.py",
    "tests/production_release_boundary_test.py",
    "tests/public_comment_contract_test.py",
    "tests/public_cli_contract_test.py",
    "tests/public_surface_contract_test.py",
    "tests/public_tree_contract_test.py",
    "tests/readme_contract_test.py",
    "tests/work_group_source_contract_test.py",
}

PRIVATE_PATH_TOKENS = (
    "/" + "home/",
    "/" + "Users/",
    "/" + "root/",
    "C:" + "\\" + "Users" + "\\",
)

INTERNAL_PATH_TOKENS = (
    "CP-" + "DIAG",
    "TP" + "6",
    "TP" + "7",
    "reviewer " + "qualification",
    "docs" + "/experiments",
    "AGENTS" + ".md",
    "CLAUDE" + ".md",
    ".vscode" + "/settings.json",
    "cmd-run" + "-results.md",
    "build-cp-" + "math-",
    "tools" + "/tp6",
    "tools" + "/tp7",
    "test-" + "datasets",
)


FORBIDDEN_PATHS = (
    "docs" + "/experiments",
    "AGENTS" + ".md",
    "CLAUDE" + ".md",
    ".vscode",
    "cmd" + ".md",
    "cmd_" + "amd.md",
    "cmd_" + "intel.md",
    "cmd-run" + "-results.md",
    "run_cmd_" + "tests.sh",
    "export_code_" + "to_md.sh",
    "tools",
    "python",
    "test-" + "datasets",
)


def tracked_files(root: Path) -> set[str]:
    if (root / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached"],
            text=True,
            capture_output=True,
            check=True,
        )
        return set(completed.stdout.splitlines())
    return set()


def is_runtime_directory(relative: Path) -> bool:
    top = relative.parts[0]
    return (
        top in {".git", ".venv", "build"}
        or top.startswith("build-")
        or "__pycache__" in relative.parts
    )


def filesystem_files(root: Path) -> set[str]:
    files: set[str] = set()
    for current, directories, filenames in os.walk(root):
        current_path = Path(current)
        directories[:] = [
            directory
            for directory in directories
            if not is_runtime_directory(
                (current_path / directory).relative_to(root)
            )
        ]
        for filename in filenames:
            relative = (current_path / filename).relative_to(root)
            if relative.suffix == ".pyc":
                continue
            files.add(relative.as_posix())
    return files


def source_files(root: Path) -> set[str]:
    return tracked_files(root) | filesystem_files(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    present_forbidden = sorted(
        relative for relative in FORBIDDEN_PATHS if (root / relative).exists()
    )
    if present_forbidden:
        raise AssertionError(
            f"forbidden private artifacts present: {present_forbidden}"
        )

    actual_files = source_files(root)
    missing = sorted(EXPECTED_FILES - actual_files)
    unexpected = sorted(actual_files - EXPECTED_FILES)
    if missing or unexpected:
        raise AssertionError(
            f"public tree mismatch; missing={missing}, unexpected={unexpected}"
        )

    for relative, expected_sha256 in EXPECTED_GENERATOR_SHA256.items():
        actual_sha256 = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise AssertionError(
                f"historical generator identity changed: {relative}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )

    for relative in sorted(PUBLIC_TEXT_FILES):
        content = (root / relative).read_text(encoding="utf-8")
        for token in PRIVATE_PATH_TOKENS + INTERNAL_PATH_TOKENS:
            if token in content:
                raise AssertionError(
                    f"publication-facing file {relative} contains private token {token}"
                )

    for relative in sorted(ACTIVE_PUBLIC_SCRIPTS):
        content = (root / relative).read_text(encoding="utf-8")
        for token in INTERNAL_PATH_TOKENS:
            if token in content:
                raise AssertionError(
                    f"public script {relative} contains internal token {token}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
