# pass arguments to ctest via command line, e.g.,  -E "logging"
set -e

mkdir -p build
cd build
cmake ..
make VERBOSE=1
ctest --output-on-failure "$@"