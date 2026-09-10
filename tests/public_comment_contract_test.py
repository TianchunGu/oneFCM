#!/usr/bin/env python3
"""Enforce English-only comments and docstrings in the public source tree."""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from pathlib import Path


CJK_RE = re.compile(
    "["
    "\\u2e80-\\u2eff"
    "\\u3000-\\u303f"
    "\\u3040-\\u30ff"
    "\\u31c0-\\u31ef"
    "\\u3400-\\u4dbf"
    "\\u4e00-\\u9fff"
    "\\uf900-\\ufaff"
    "]"
)

STALE_PATTERNS = (
    "keep unchanged",
    "omitted for brevity",
    "compared with the old code",
    "compared with the original code",
)

CPP_SUFFIXES = {".cpp", ".h"}
CMAKE_FILES = (
    "CMakeLists.txt",
    "cmeans/CMakeLists.txt",
    "util/CMakeLists.txt",
    "tests/CMakeLists.txt",
)


def cpp_comments(text: str) -> list[tuple[int, str]]:
    """Extract C/C++ comments without interpreting quoted or raw strings."""
    comments: list[tuple[int, str]] = []
    i = 0
    line = 1
    while i < len(text):
        if text.startswith("//", i):
            start = i + 2
            end = text.find("\n", start)
            if end < 0:
                end = len(text)
            comments.append((line, text[start:end]))
            i = end
            continue
        if text.startswith("/*", i):
            start_line = line
            end = text.find("*/", i + 2)
            if end < 0:
                raise AssertionError(
                    f"unterminated C/C++ block comment at line {start_line}"
                )
            body = text[i + 2 : end]
            comments.append((start_line, body))
            line += body.count("\n")
            i = end + 2
            continue
        raw_match = re.match(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(', text[i:])
        if raw_match:
            terminator = ")" + raw_match.group(1) + '"'
            start = i + raw_match.end()
            end = text.find(terminator, start)
            if end < 0:
                raise AssertionError(f"unterminated raw string at line {line}")
            literal = text[i : end + len(terminator)]
            line += literal.count("\n")
            i = end + len(terminator)
            continue
        quote_match = re.match(r'(?:u8|u|U|L)?(["\'])', text[i:])
        if quote_match:
            quote = quote_match.group(1)
            i += quote_match.end()
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    line += (text[i] + text[i + 1]).count("\n")
                    i += 2
                    continue
                char = text[i]
                i += 1
                if char == "\n":
                    line += 1
                if char == quote:
                    break
            continue
        if text[i] == "\n":
            line += 1
        i += 1
    return comments


def python_documentation(path: Path) -> list[tuple[int, str]]:
    """Extract Python comments plus module, class, and function docstrings."""
    text = path.read_text(encoding="utf-8")
    found = [
        (token.start[0], token.string[1:].strip())
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.COMMENT
    ]
    tree = ast.parse(text, filename=str(path))
    nodes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, nodes) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.append((first.lineno, first.value.value))
    return found


def cmake_comments(text: str) -> list[tuple[int, str]]:
    """Extract CMake comments outside quoted and bracket arguments."""
    comments: list[tuple[int, str]] = []
    i = 0
    line = 1
    while i < len(text):
        if text[i] == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    line += (text[i] + text[i + 1]).count("\n")
                    i += 2
                    continue
                char = text[i]
                i += 1
                if char == "\n":
                    line += 1
                if char == '"':
                    break
            continue
        bracket = re.match(r"\[(=*)\[", text[i:])
        if bracket:
            terminator = "]" + bracket.group(1) + "]"
            start = i + bracket.end()
            end = text.find(terminator, start)
            if end < 0:
                raise AssertionError(f"unterminated CMake bracket argument at line {line}")
            argument = text[i : end + len(terminator)]
            line += argument.count("\n")
            i = end + len(terminator)
            continue
        if text[i] == "#":
            start = i + 1
            end = text.find("\n", start)
            if end < 0:
                end = len(text)
            comments.append((line, text[start:end]))
            i = end
            continue
        if text[i] == "\n":
            line += 1
        i += 1
    return comments


def public_source_files(root: Path) -> list[Path]:
    paths: set[Path] = {root / relative for relative in CMAKE_FILES}
    for directory in ("cmeans", "util", "tests"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix in CPP_SUFFIXES:
                paths.add(path)
    paths.update(path for path in (root / "tests").rglob("*.py") if path.is_file())
    return sorted(paths)


def documentation_for(root: Path, path: Path) -> list[tuple[int, str]]:
    relative = path.relative_to(root).as_posix()
    if relative in CMAKE_FILES:
        return cmake_comments(path.read_text(encoding="utf-8"))
    if path.suffix in CPP_SUFFIXES:
        return cpp_comments(path.read_text(encoding="utf-8"))
    if path.suffix == ".py":
        return python_documentation(path)
    raise AssertionError(f"unsupported public source type: {relative}")


def source_region(text: str, start: str, end: str) -> str:
    """Return a source region delimited by stable declaration fragments."""
    try:
        region_start = text.index(start)
        region_end = text.index(end, region_start + len(start))
    except ValueError as error:
        raise AssertionError(
            f"comment placement boundary missing: start={start!r}, end={end!r}"
        ) from error
    return text[region_start:region_end]


def check_classification_comment_placement(root: Path) -> list[str]:
    """Guard the known change-metric and classification comment placement."""
    source = (root / "cmeans/src/cmeans.cpp").read_text(encoding="utf-8")
    change_region = source_region(source, "float change_matrix(", "void classify(")
    classify_region = source_region(
        source,
        "void classify(",
        "template <typename CenterPowerEvaluator, typename MembershipPowerEvaluator>",
    )
    change_comments = " ".join(
        comment.strip() for _, comment in cpp_comments(change_region)
    ).lower()
    classify_comments = " ".join(
        comment.strip() for _, comment in cpp_comments(classify_region)
    ).lower()

    failures: list[str] = []
    if "maximum absolute element-wise change" not in change_comments:
        failures.append("change_matrix comment does not describe its reduction metric")
    for misplaced in (
        "maximum membership",
        "one-based cluster label",
        "one-based labels",
        "expose one-based",
    ):
        if misplaced in change_comments:
            failures.append(
                f"change_matrix contains classification wording: {misplaced!r}"
            )
    for required in (
        "select the maximum membership for each point",
        "one-based cluster",
        "1..k",
    ):
        if required not in classify_comments:
            failures.append(f"classify comment missing semantic fragment: {required!r}")
    if "label = i + 1;" not in classify_region:
        failures.append("classify no longer emits one-based labels with label = i + 1")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    failures: list[str] = []
    files = public_source_files(root)
    for path in files:
        relative = path.relative_to(root).as_posix()
        for line, comment in documentation_for(root, path):
            if CJK_RE.search(comment):
                failures.append(f"{relative}:{line}: CJK source documentation")
            lowered = " ".join(comment.lower().split())
            for pattern in STALE_PATTERNS:
                if pattern in lowered:
                    failures.append(
                        f"{relative}:{line}: stale comment phrase: {pattern!r}"
                    )

    failures.extend(check_classification_comment_placement(root))

    if failures:
        raise AssertionError("public comment contract failed:\n" + "\n".join(failures))

    print(f"public comment contract passed: files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
