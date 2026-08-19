Uses and requires the [catch2](https://github.com/catchorg/Catch2) framework for testing, at least at version 3.4.
Under ubuntu, this can be installed via apt. For other OS (e.g. Debian11), one can install a recent version of catch2 via conda.
Uses and requires [xtensor](https://github.com/xtensor-stack/xtensor) and [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) for the cpp tensor representations, both can be installed via conda
Uses and requires [valgrind](https://valgrind.org/) for memory leak checking, can be installed via apt.

Memory leaks inside a catch2 testcase are not caught by valgrind.
Therefore, we have a separate, independent executable that checks if memory is handled appropriately across the rust-C interface.

Tests are split into two cmake projects because catch2 isn't compatible with the default version of gcc on debian11.
Therefore, we omit unit tests on debian (functionality of the toolkit package is checked in the ubuntu CI).
We still want to execute a program based on the toolkit on debian11 to see "if it runs", we use the memory test for this.
