set -e

# bring .h files in the same folder such that the resulting file tree in the 
# html output does not show any file structure/names that are only present in 
# the source code version
cp ../../src/c_bridge/*.h .
doxygen
rm -r ./*.h