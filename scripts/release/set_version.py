#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_FILE = Path(__file__).resolve().parents[2] / "lightrag" / "_version.py"


def normalize_core_version(raw_version: str) -> str:
    if raw_version.startswith("v") and len(raw_version) > 1:
        return raw_version[1:]
    return raw_version


def read_public_version(content: str) -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]*)"$', content, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read __version__ in {VERSION_FILE}")
    return match.group(1).split("+", 1)[0]


def update_assignment(content: str, name: str, value: str) -> str:
    pattern = rf'^{name}\s*=\s*"[^"]*"$'
    updated, count = re.subn(
        pattern,
        f'{name} = "{value}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"Could not update {name} in {VERSION_FILE}")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Update LightRAG version constants.")
    parser.add_argument(
        "--core-version",
        help="Replace __version__ entirely. A leading 'v' is stripped automatically.",
    )
    parser.add_argument(
        "--append-local",
        help="Keep the public __version__ from the file and append +LOCAL (git tag).",
    )
    parser.add_argument(
        "--api-version",
        help="Optional API compatibility version override.",
    )
    args = parser.parse_args()

    if bool(args.core_version) == bool(args.append_local):
        parser.error("Provide exactly one of --core-version or --append-local")

    content = VERSION_FILE.read_text(encoding="utf-8")
    if args.append_local:
        local = normalize_core_version(args.append_local)
        core_version = f"{read_public_version(content)}+{local}"
    else:
        core_version = normalize_core_version(args.core_version)
    content = update_assignment(content, "__version__", core_version)
    if args.api_version is not None:
        content = update_assignment(content, "__api_version__", args.api_version)

    VERSION_FILE.write_text(content, encoding="utf-8")
    print(core_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
