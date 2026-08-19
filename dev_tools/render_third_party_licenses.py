"""Render THIRD_PARTY_LICENSES.md from `cargo about generate --format json` output.

`cargo about` collects the license data but its Handlebars templates cannot sort.
This renderer produces the same document with the summary table flattened and every listing
sorted alphabetically.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

# This crate itself appears in the graph; it is not a third-party dependency.
SELF_CRATE = "qant_native_computing_toolkit"

DOCUMENT_HEADER = """# Third Party Licenses

This file lists the licenses of the third-party dependencies bundled with this
project.

## Summary

The following table lists all bundled dependencies and their license type. The
full license texts are reproduced in the sections below.

| Dependency | License |
| --- | --- |"""


def iter_third_party_crates(license_entry: dict) -> Iterator[dict]:
    """Yield the crates covered by a license entry, skipping this project's own crate."""
    for usage in license_entry["used_by"]:
        crate = usage["crate"]
        if crate["name"] != SELF_CRATE:
            yield crate


def format_crate_link(crate: dict) -> str:
    """Return the crate as a Markdown link to its repository, or its bare name."""
    repository = crate.get("repository")
    if not repository:
        return crate["name"]
    return f"[{crate['name']}]({repository})"


def crate_sort_key(crate: dict) -> str:
    """Return the key that orders crates alphabetically, ignoring case."""
    return crate["name"].lower()


def collect_summary_rows(licenses: list[dict]) -> list[tuple[str, str, str]]:
    """Return (name, link, license) rows sorted alphabetically by dependency name."""
    rows = {
        (crate["name"], format_crate_link(crate), license_entry["name"])
        for license_entry in licenses
        for crate in iter_third_party_crates(license_entry)
    }
    return sorted(rows, key=lambda row: (row[0].lower(), row))


def collect_license_sections(licenses: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """Return (license name, license text, crates) sections, sorted alphabetically.

    Sections are ordered by license name and then by the crates they cover, and the
    crates within a section are ordered by name.
    """
    sections = []
    for license_entry in licenses:
        crates = sorted(iter_third_party_crates(license_entry), key=crate_sort_key)
        if crates:
            sections.append((license_entry["name"], license_entry["text"], crates))
    return sorted(
        sections,
        # the same license name (section[0]) appears many times, e.g. for apache-2.
        # In this case, break the tie by sorting by crate name (section[2])
        key=lambda section: (section[0], list(map(crate_sort_key, section[2]))),
    )


def render_markdown(licenses: list[dict]) -> str:
    """Render the full THIRD_PARTY_LICENSES.md document from the cargo-about entries."""
    lines = [DOCUMENT_HEADER]
    lines.extend(
        f"| {link} | {license_name} |"
        for _, link, license_name in collect_summary_rows(licenses)
    )
    lines.append("")

    for license_name, license_text, crates in collect_license_sections(licenses):
        lines.extend(["---", "", f"## {license_name}", "", "Used by:", ""])
        lines.extend(f"- {format_crate_link(crate)}" for crate in crates)
        lines.extend(["", "```", license_text.rstrip("\n"), "```", ""])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render THIRD_PARTY_LICENSES.md from cargo-about JSON output."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSON file written by `cargo about generate --format json`",
    )
    parser.add_argument("output", type=Path, help="Markdown file to write")
    args = parser.parse_args()

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read {args.input}: {error}")

    args.output.write_text(render_markdown(data["licenses"]), encoding="utf-8")


if __name__ == "__main__":
    main()
