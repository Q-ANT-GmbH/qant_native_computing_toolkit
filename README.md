# Q.ANT native computing toolkit

This library provides the official API for running workloads on Q.ANT native processing units.

## Installation

### Prerequisites

If you have access to Q.ANT hardware, you must have the corresponding drivers installed.
If you want to use the Q.ANT SDK in standalone mode on CPU, Q.ANT drivers are not required.

### Installation from release artifacts

You find the required files in the release section on this github repo.
The packages come in two versions:

- default: Require Q.ANT drivers and hardware
- cpu-backend: Standalone version for development without Q.ANT hardware

| Language | Installation tool |
| -------- | ------------ |
| Python   | Install the package wheel for your python version via ``pip install qant_native_computing_toolkit.<version>.whl`` |
| C/C++    | Install the Debian package via ``apt install ./qant_native_computing_toolkit.<version>.deb`` |

### Installation from source

This repository uses git submodules, which must be checked out before building. Either clone with `git clone --recurse-submodules <repo-url>`, or run `git submodule update --init --recursive` after a plain clone.

| Language | Installation |
| -------- | ------------ |
| Python   | If you have Q.ANT hardware, install the package via ``pip install .``. If you want to use the SDK in standalone mode, ``pip install maturin`` and install the package via ``maturin develop -F cpu-backend`` |
| C/C++    | Build the shared library with ``cargo build --release``. Pass ``-F cpu-backend`` for standalone mode. Use [create_deb.sh](src/c_bridge/packaging/create_deb.sh) to build the debian package. Then install via ``apt install ./target/deb/qant-native-computing-toolkit.<version>.deb``|

For more details, see the respective workflows [`workflow_python.yml`](.github/workflows/workflow_python.yml) and [`workflow_c.yml`](.github/workflows/workflow_c.yml).

## Documentation

Documentation for the latest release can be found [here](https://q-ant-gmbh.github.io/qant_native_computing_toolkit/)

## Examples

A collection of examples can be found in this separate [repo](https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit_examples).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports, feature requests, the PR process, and how to build and test the project.
