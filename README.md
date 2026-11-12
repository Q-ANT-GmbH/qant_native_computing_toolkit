# Q.ANT native computing toolkit

This library provides the official API for running workloads on Q.ANT native processing units.

## Installation

### Prerequisites

If you have access to Q.ANT hardware, you must have the corresponding drivers installed.
If you want to use the Q.ANT SDK in standalone mode on CPU, Q.ANT drivers are not required.

### Supported platforms

**Python:** 3.11, 3.12, 3.13 and 3.14.

| Variant     | Linux x86_64 | Windows x86_64 | macOS arm64 |
| ----------- | ------------ | -------------- | ----------- |
| default     | ✅           | –              | –           |
| cpu-backend | ✅           | ✅             | ✅          |

Linux wheels are built on Debian 13 and tested on Debian 13 and Ubuntu 24.04.

### Installation from release artifacts

You find the required files in the release section on this github repo.
The packages come in two versions:

- default: Require Q.ANT drivers and hardware
- cpu-backend: Standalone version for development without Q.ANT hardware

There is one Python wheel per OS and variant, and it works with every supported Python version.
The `cp311-abi3` tag in its file name means "Python 3.11 or newer", so pick the wheel for your OS:
`linux_x86_64`, `win_amd64` or `macosx_..._arm64`.

| Language | Installation tool |
| -------- | ------------ |
| Python   | Install the wheel for your OS, e.g. ``pip install qant_native_computing_toolkit-<version>-cp311-abi3-linux_x86_64.whl`` |
| C/C++    | Install the Debian package via ``apt install ./qant_native_computing_toolkit.<version>.deb`` |

### Installation from source

This repository uses git submodules, which must be checked out before building. Either clone with `git clone --recurse-submodules <repo-url>`, or run `git submodule update --init --recursive` after a plain clone.

| Language | Installation |
| -------- | ------------ |
| Python   | If you have Q.ANT hardware, install the package via ``pip install .``. If you want to use the SDK in standalone mode, ``pip install maturin`` and install the package via ``maturin develop -F cpu-backend`` |
| C/C++    | Build the shared library with ``cargo build --release``. Pass ``-F cpu-backend`` for standalone mode. Use [create_deb.sh](src/c_bridge/packaging/create_deb.sh) to build the debian package. Then install via ``apt install ./target/deb/qant-native-computing-toolkit.<version>.deb``|

Building from source needs a toolchain per language. [CONTRIBUTING.md](CONTRIBUTING.md#setting-up-a-local-development-environment) documents the full setup, including how to build and test without Q.ANT hardware.

## Documentation

Documentation for the latest release can be found [here](https://q-ant-gmbh.github.io/qant_native_computing_toolkit/)

## Examples

A collection of examples can be found in this separate [repo](https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit_examples).

## Community

Please join us in the [Discussions](https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit/discussions/) section of this repo for announcements and Q&A.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports, feature requests, the PR process, and how to build and test the project.
