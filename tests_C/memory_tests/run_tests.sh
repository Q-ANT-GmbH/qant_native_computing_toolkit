set -e

# If you encounter memory leaks, build with this profile to get debug information
# cargo build --profile=release-with-debug --no-default-features
mkdir -p build
cd build
cmake ..
make VERBOSE=1
echo
echo "build done, executing tests..."
valgrind --leak-check=full --show-leak-kinds=all --show-reachable=no --error-exitcode=1 ./test_memory_safety
echo 
valgrind --leak-check=full --show-leak-kinds=all --show-reachable=no --error-exitcode=1 ./test_memory_safety_i16
echo 
valgrind --leak-check=full --show-leak-kinds=all --show-reachable=no --error-exitcode=1 ./test_memory_safety_mat
