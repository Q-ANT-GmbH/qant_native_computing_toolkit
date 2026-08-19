#!/bin/bash
set -e
SELF_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(realpath "$SELF_DIR/..")

if ! command -v tq &> /dev/null; then
    echo "Error: Required command 'tq' not found. Install tomlq via: cargo install tomlq" >&2
    exit 1
fi

VERSION=$(tq -r -f "$REPO_ROOT/Cargo.toml" package.version)
echo "Current Cargo.toml Version: $VERSION"
read -p "New Version: " VERSION

echo "New Version: $VERSION"
IFS='.' read -r MAJOR MINOR _ <<< "$VERSION"

echo "Adapt all locations.."
echo "Update Cargo.toml"
sed -i -e "s/^version = \".*\"/version = \"$VERSION\"/" "$REPO_ROOT/Cargo.toml"

echo "Update pyproject.toml"
sed -i -e "s/version = \".*\"/version = \"$VERSION\"/" "$REPO_ROOT/pyproject.toml"

echo "Update C API"
sed -i -e "s/#define QANT_NATIVE_COMPUTING_TOOLKIT_MAJOR_VERSION.*/#define QANT_NATIVE_COMPUTING_TOOLKIT_MAJOR_VERSION $MAJOR/" "$REPO_ROOT/src/c_bridge/qant_native_computing_toolkit_info.h"
sed -i -e "s/#define QANT_NATIVE_COMPUTING_TOOLKIT_MINOR_VERSION.*/#define QANT_NATIVE_COMPUTING_TOOLKIT_MINOR_VERSION $MINOR/" "$REPO_ROOT/src/c_bridge/qant_native_computing_toolkit_info.h"

echo "Update C API Docs (Doxygen)"
sed -i -e "s/PROJECT_NUMBER\s*=.*/PROJECT_NUMBER         = $VERSION/" "$REPO_ROOT/doc/doxygen/Doxyfile"

echo "Update Python API"
sed -i -e "s/__version__\s*=.*/__version__ = \"$VERSION\"/" "$REPO_ROOT/python/qant_native_computing_toolkit/__init__.py"

echo "Update Python Docs"
sed -i -e "s/release\s*=.*/release = \"$VERSION\"/" "$REPO_ROOT/doc/python/source/conf.py"

echo "done."
echo ""

echo "Please add a changelog entry for the C API package.."
read -p "Favorite editor (e.g., vim, nano): " editor
$editor "$REPO_ROOT/src/c_bridge/packaging/changelog"



