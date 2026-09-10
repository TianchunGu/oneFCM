#!/usr/bin/env python3
"""Protect the membership-initialization timing boundary."""

from __future__ import annotations

import argparse
from pathlib import Path


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    body = function_body(source, "parallel_initialize_membership_matrix(")

    submit = body.index("sycl::event event = q.submit")
    wait = body.index("event.wait_and_throw();")
    profiling_guard = body.index("#if FCM_ENABLE_PROFILING")
    profiling_measurement = body.index("get_kernel_execution_time(event)")
    returned_event = body.index("return event;")

    assert submit < wait < profiling_guard < profiling_measurement < returned_event
    assert "event.wait();" not in body
    assert body.count("event.wait_and_throw();") == 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
