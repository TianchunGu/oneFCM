#!/usr/bin/env python3
"""Guard the public Platform A container provenance and wording contracts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


UR_COMMIT = "d040a8face402e5ecc7c99cdb4e976832592ce99"
UMF_COMMIT = "59c4150b7120a7af5b3c8eb2d9b8bbb5d2e96aa3"


def require(text: str, fragments: tuple[str, ...], artifact: str) -> None:
    for fragment in fragments:
        if fragment not in text:
            raise AssertionError(f"{artifact} contract missing: {fragment}")


def docker_pattern_regex(pattern: str) -> re.Pattern[str]:
    """Compile the Docker-style glob subset used by this public context."""
    cleaned = pattern.strip("/")
    has_slash = "/" in cleaned
    expression: list[str] = []
    index = 0
    while index < len(cleaned):
        if cleaned.startswith("**/", index):
            expression.append("(?:.*/)?")
            index += 3
        elif cleaned.startswith("**", index):
            expression.append(".*")
            index += 2
        elif cleaned[index] == "*":
            expression.append("[^/]*")
            index += 1
        elif cleaned[index] == "?":
            expression.append("[^/]")
            index += 1
        else:
            expression.append(re.escape(cleaned[index]))
            index += 1
    body = "".join(expression)
    if not has_slash:
        body = "(?:.*/)?" + body
    return re.compile(rf"^{body}(?:/.*)?$")


def context_includes(path: str, patterns: tuple[str, ...]) -> bool:
    """Evaluate ordered ignore and negation rules for a controlled path."""
    included = True
    for raw_pattern in patterns:
        negated = raw_pattern.startswith("!")
        pattern = raw_pattern[1:] if negated else raw_pattern
        if docker_pattern_regex(pattern).fullmatch(path.strip("/")):
            included = negated
    return included


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    dockerfile = (
        root / "containers/Dockerfile.platform-a"
    ).read_text(encoding="utf-8")
    readme = (root / "containers/README.md").read_text(encoding="utf-8")
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    dockerignore_patterns = tuple(
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    require(
        dockerignore,
        (
            ".venv",
            "**/__pycache__",
            "**/*.pyc",
            "!tests/fixtures/tiny.csv",
        ),
        ".dockerignore",
    )
    excluded_runtime_paths = (
        ".venv",
        ".venv/bin/python",
        "__pycache__",
        "foo/__pycache__",
        "foo/__pycache__/module.cpython-312.pyc",
        "module.pyc",
        "foo/bar/module.pyc",
    )
    unexpectedly_included = tuple(
        path
        for path in excluded_runtime_paths
        if context_includes(path, dockerignore_patterns)
    )
    if unexpectedly_included:
        raise AssertionError(
            "runtime artifacts remain in Docker context: "
            f"{unexpectedly_included}"
        )
    if not context_includes("tests/fixtures/tiny.csv", dockerignore_patterns):
        raise AssertionError("tiny.csv fixture is excluded from Docker context")

    require(
        dockerfile,
        (
            "ARG ONEFCM_SOURCE_COMMIT=unknown",
            'test -n "${ONEFCM_SOURCE_COMMIT}"',
            'test "${ONEFCM_SOURCE_COMMIT}" != "unknown"',
            '"onefcm_source_commit=${ONEFCM_SOURCE_COMMIT}"',
            'org.opencontainers.image.revision="${ONEFCM_SOURCE_COMMIT}"',
            'org.opencontainers.image.source="https://github.com/TianchunGu/oneFCM"',
            f"ARG UR_COMMIT={UR_COMMIT}",
            f"ARG UMF_COMMIT={UMF_COMMIT}",
            '"unified_runtime_commit=${UR_COMMIT}"',
            '"unified_memory_framework_commit=${UMF_COMMIT}"',
            "pinned Unified Runtime v0.10.0 revision",
            "pinned UMF source supplied locally",
            "source download",
        ),
        "Dockerfile",
    )
    positions = [
        dockerfile.index(fragment)
        for fragment in (
            "ARG ONEFCM_SOURCE_COMMIT=unknown",
            'test -n "${ONEFCM_SOURCE_COMMIT}"',
            'test "${ONEFCM_SOURCE_COMMIT}" != "unknown"',
            '"onefcm_source_commit=${ONEFCM_SOURCE_COMMIT}"',
            'org.opencontainers.image.revision="${ONEFCM_SOURCE_COMMIT}"',
        )
    ]
    if positions != sorted(positions):
        raise AssertionError("source-commit hard gate does not precede metadata use")

    require(
        readme,
        (
            'test -z "$(git status --short)"',
            'ONEFCM_SOURCE_COMMIT="$(git rev-parse HEAD)"',
            '--build-arg ONEFCM_SOURCE_COMMIT="${ONEFCM_SOURCE_COMMIT}"',
            'test "$(git -C unified-runtime rev-parse HEAD)" =',
            UR_COMMIT,
            'test -z "$(git -C unified-runtime status --short)"',
            'test "$(git -C unified-memory-framework rev-parse HEAD)" =',
            UMF_COMMIT,
            'test -z "$(git -C unified-memory-framework status --short)"',
            "the Dockerfile does not inspect Git metadata in",
            "## Performance validation",
            "84 container",
            "approximately\n0.31%",
            "below 0.87%",
            "no systematic\nslowdown",
            "not a paired measurement of strict\ncontainerization overhead",
        ),
        "container README",
    )
    readme_positions = [
        readme.index(fragment)
        for fragment in (
            'test -z "$(git status --short)"',
            'ONEFCM_SOURCE_COMMIT="$(git rev-parse HEAD)"',
            "DOCKER_BUILDKIT=1 docker build",
        )
    ]
    if readme_positions != sorted(readme_positions):
        raise AssertionError("oneFCM clean/source gates must precede docker build")

    for forbidden in (
        "BuildKit secrets are limited",
        "500 KiB",
        "Docker overhead = 0.31%",
        "negligible Docker overhead",
        "Docker has zero performance impact",
        "exactly reproduces native performance",
        "byte-identical to historical Codeplay binary",
        "ABI-compatible source line used by the Intel",
        "https://github.com/TianchunGu/fuzzy-c-means-clustering-based-on-SYCL-and-oneAPI",
    ):
        if re.search(re.escape(forbidden), readme + dockerfile, re.IGNORECASE):
            raise AssertionError(f"forbidden container wording: {forbidden}")

    print("container contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
