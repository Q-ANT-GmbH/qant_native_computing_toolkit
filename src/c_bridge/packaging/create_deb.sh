# adapted from https://www.internalpointers.com/post/build-binary-deb-package-practical-guide

#
# takes one argument: mode (optional, string). 
# "default": (default) build for deployment on real device
# "cpu-backend": build for cpu backend
set -e

mode=${1:-default}
if [ "$mode" == "default" ]; then
    DRIVER="qant-native-computing-driver"
elif [ "$mode" == "cpu-backend" ]; then
    echo ""
else
    echo "Invalid argument: $mode"
    echo "Usage: $0 [cpu-backend|default]"
    exit 1
fi

DEPENDENCIES_LIST=(
    "libc6"
)

if [ -n "$DRIVER" ]; then
    DEPENDENCIES_LIST+=("$DRIVER")
fi


SELF_DIR=$(dirname "$(realpath "$0")")
RUST_ROOT=$SELF_DIR/../../..

# general metadata, will be used multiple times later
VERSION="$(python3 "$RUST_ROOT/dev_tools/read_toml_version.py" "$RUST_ROOT/Cargo.toml" package)"
EMAIL="info@qant.de"
AUTHOR="Q.ANT GmbH"
PKG_NAME="qant-native-computing-toolkit"
YEAR="2026"
DEPENDENCIES=$(IFS=, ; echo "${DEPENDENCIES_LIST[*]}")
DEBUG="false"

# find the files we want to package
if [ "$DEBUG" =  "false" ]; then
    BUILD_MODE="release"
else 
    BUILD_MODE="debug"
fi


# find the files we want to package
LIB_FILE="$RUST_ROOT/target/$BUILD_MODE/libqant_native_computing_toolkit.so"
H_FILES=("$RUST_ROOT"/src/c_bridge/qant_native_computing_toolkit*.h)
LIB_NAME="libqant_native_computing_toolkit"
SO_NAME="${LIB_NAME}.so"
THIRD_PARTY_LICENSES_FNAME="THIRD_PARTY_LICENSES.md"

# collect information about host system (the system this .deb gets packaged for)
MULTIARCHITECTURE=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
ARCHITECTURE=$(dpkg-architecture -qDEB_HOST_ARCH)

# where the files needed for deb creation are put
# we build the package in the /tmp-directory as here we do not have to handle special user rights
DEB_ROOT="/tmp/qant_build/$PKG_NAME.$VERSION-1-$ARCHITECTURE"
# where to place the final .deb
OUT_DIR=$RUST_ROOT/target/deb/

# dir where the .so will be placed
SO_DIR="usr/lib/$MULTIARCHITECTURE"
# dir where the .h will be placed
INCL_DIR="/usr/include/$MULTIARCHITECTURE"
# dir where the docs (license, changelog, ...) will be placed
DOC_DIR="usr/share/doc/$PKG_NAME"

set -e

mkdir -p "$DEB_ROOT"
mkdir -p "$DEB_ROOT/$SO_DIR"
mkdir -p "$DEB_ROOT/$INCL_DIR"
mkdir -p "$DEB_ROOT/DEBIAN"
mkdir -p "$DEB_ROOT/$DOC_DIR"
mkdir -p "$OUT_DIR"

# copy .so to target location and strip debug symbols
cp "$LIB_FILE" "$DEB_ROOT/$SO_DIR"
if [ "$DEBUG" =  "false" ]; then
	strip --strip-all "$DEB_ROOT/$SO_DIR/$SO_NAME"
fi

# rename .so to include version
SO_NAME_VERSIONED="${SO_NAME}.${VERSION}"
mv "$DEB_ROOT/$SO_DIR/$SO_NAME" "$DEB_ROOT/$SO_DIR/$SO_NAME_VERSIONED"
# rename the internal SONAME to match the file name
patchelf --set-soname "$SO_NAME_VERSIONED" "$DEB_ROOT/$SO_DIR/$SO_NAME_VERSIONED"
# remove exec rights
chmod 644 "$DEB_ROOT/$SO_DIR/$SO_NAME_VERSIONED"

# copy the headers to their target location
cp "${H_FILES[@]}" "$DEB_ROOT/$INCL_DIR"
# add read access
chmod a+r "$DEB_ROOT/$INCL_DIR"/*.h

# generate control file to set all the metadata 
# (important: the dependencies will be enforced at install time)
cat > "$DEB_ROOT/DEBIAN/control" << EOF
Package: $PKG_NAME
Version: $VERSION
Architecture: $ARCHITECTURE
Maintainer: $AUTHOR <$EMAIL>
Priority: optional
Section: devel
Depends: $DEPENDENCIES
Description: Toolkit for the Q.ANT native processing unit. 
 Contains the $SO_NAME shared library 
 and corresponding header files.
EOF


cat > "$DEB_ROOT/usr/share/doc/$PKG_NAME/copyright" << EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: qant native computing toolkit
Files:
 *
Copyright: $YEAR $AUTHOR <$EMAIL>
License: Apache-2.0

Comment:
 $SO_NAME statically links third-party dependencies. 
 The full text of their licenses is distributed next to this
 file:
 .
   $THIRD_PARTY_LICENSES_FNAME
   LICENSE_dlpack.txt
EOF

# create trigger file to automatically ldconfig when the .deb is installed
cat > "$DEB_ROOT/DEBIAN/triggers" << EOF
activate-noawait ldconfig
EOF

# declare which shared libraries this package provides
cat > "$DEB_ROOT/DEBIAN/shlibs" << EOF
$LIB_NAME    $VERSION    $SONAME
EOF

# create symlink to also have unversioned file that the linker finds
cat > "$DEB_ROOT/DEBIAN/postinst" << EOF
#!/bin/bash

# script must abort on any error to ensure proper cleanup
set -e
# old name: remove symbolic link
if [ -L /$SO_DIR/lib$PKG_NAME.so ]; then
    echo "Remove old .so symbolic link"
    rm /$SO_DIR/lib$PKG_NAME.so
fi

# symbolic link should always point to the latest installed version
if [ -L /$SO_DIR/$SO_NAME ]; then
    echo "Update .so symbolic link"
    rm /$SO_DIR/$SO_NAME
fi

ln -s /$SO_DIR/$SO_NAME_VERSIONED /$SO_DIR/$SO_NAME

EOF
chmod ugo+x $DEB_ROOT/DEBIAN/postinst

# bundle the licenses of the third-party code that is statically linked into
# $SO_NAME
cp $RUST_ROOT/$THIRD_PARTY_LICENSES_FNAME $DEB_ROOT/$DOC_DIR/$THIRD_PARTY_LICENSES_FNAME
cp $RUST_ROOT/external/dlpack/LICENSE $DEB_ROOT/$DOC_DIR/LICENSE_dlpack.txt

# compress and copy changelog to target location
# need to use -9 for maximum compression
# --no-name to not store the original file name and the timestamp
gzip --no-name -9 -c "$SELF_DIR/changelog" > "$DEB_ROOT/$DOC_DIR/changelog.gz"

# This command finally builds the .deb
dpkg-deb --build --root-owner-group "$DEB_ROOT"

# Move the .deb to the target dir
mv "$DEB_ROOT"/../*.deb "$OUT_DIR"

# delete artifacts
rm -r "$DEB_ROOT"

# lintian checks the .deb for common mistakes
lintian --pedantic --verbose --fail-on error "$OUT_DIR"/*.deb
