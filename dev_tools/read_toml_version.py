"""Print the `version` field of a top-level table in a TOML file.

Cargo.toml (`[package]`) and pyproject.toml (`[project]`) are the sources of the
release version, and both CI and the release tooling need to read them from shell.
python3 ships tomllib, so no extra TOML reader has to be installed.
"""

import argparse
import tomllib
from pathlib import Path


def read_version(toml_file: Path, table: str) -> str:
    """Return the `version` of the given top-level table in the TOML file."""
    with toml_file.open("rb") as handle:
        return tomllib.load(handle)[table]["version"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the version of a top-level table in a TOML file."
    )
    parser.add_argument(
        "toml_file", type=Path, help="TOML file to read, e.g. Cargo.toml"
    )
    parser.add_argument(
        "table", help="top-level table holding the version, e.g. package or project"
    )
    args = parser.parse_args()

    try:
        print(read_version(args.toml_file, args.table))
    except (OSError, tomllib.TOMLDecodeError) as error:
        parser.error(f"cannot read {args.toml_file}: {error}")
    except KeyError as error:
        parser.error(f"{args.toml_file} has no {args.table}.version: missing {error}")


if __name__ == "__main__":
    main()
