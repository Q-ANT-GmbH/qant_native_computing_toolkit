Uses and requires the [catch2](https://github.com/catchorg/Catch2) framework for testing and [xtensor](https://github.com/xtensor-stack/xtensor) plus [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) for the cpp tensor representations.
Those are pinned in [`environment.yml`](../environment.yml) in the repository root and are picked up from `$CONDA_PREFIX`, so that conda environment must be active when running the tests — see [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the setup.
Uses and requires [valgrind](https://valgrind.org/) for memory leak checking, can be installed via apt.

Memory leaks inside a catch2 testcase are not caught by valgrind.
Therefore, we have a separate, independent executable that checks if memory is handled appropriately across the rust-C interface.

Tests are split into two cmake projects because catch2 isn't compatible with the default version of gcc on debian11.
Therefore, we omit unit tests on debian (functionality of the toolkit package is checked in the ubuntu CI).
We still want to execute a program based on the toolkit on debian11 to see "if it runs", we use the memory test for this.
