#!/bin/bash
# Regenerate THIRD_PARTY_LICENSES.md from the current dependency graph.
#
# `cargo about` gathers the license data, but its Handlebars templates cannot sort,
# so it emits JSON here and render_third_party_licenses.py renders the document.
set -e

SELF_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(realpath "$SELF_DIR/..")
OUTPUT=$REPO_ROOT/THIRD_PARTY_LICENSES.md

if ! command -v cargo-about &> /dev/null; then
    echo "Error: Required command 'cargo-about' not found. Install via: cargo install cargo-about --version 0.9.0" >&2
    exit 1
fi

# temporary file to hold the json. Clean up when script finishes.
LICENSES_JSON=$(mktemp)
trap 'rm -f "$LICENSES_JSON"' EXIT

cargo about generate \
    --all-features \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    -c "$SELF_DIR/about.toml" \
    --format json \
    -o "$LICENSES_JSON"

python3 "$SELF_DIR/render_third_party_licenses.py" "$LICENSES_JSON" "$OUTPUT"

echo "Wrote $OUTPUT"
