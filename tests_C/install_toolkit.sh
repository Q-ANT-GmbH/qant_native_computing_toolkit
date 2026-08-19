set -e

SELF_DIR=$(dirname "$(realpath "$0")")
RUST_ROOT=$SELF_DIR/../

BUILD_PROFILE="${1:-release}"
# choose build option. Default: release
# for debugging, release-with-debug does not strip symbols from the .so

cargo build --no-default-features --profile=$BUILD_PROFILE
$RUST_ROOT/src/c_bridge/packaging/create_deb.sh
sudo apt install --reinstall $RUST_ROOT/target/deb/qant-native-computing-toolkit*.deb
echo "toolkit successfully installed"