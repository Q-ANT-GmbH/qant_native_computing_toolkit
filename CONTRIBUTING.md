# Contributing

## Bug reports and feature requests

Please use the issues section of this repo to file bug reports and feature requests.

## Pull requests

You are welcome to submit PRs to this repository.
If you plan to implement a new feature, please get in contact with our dev team so we can align the changes with our roadmap.

Please fork the repo and submit the PR from there, creating branches on this repo directly is not allowed for people outside the Q.ANT organisation.
The PR will be reviewed by our team and accepted changes will be included in the next official release, where your contributions will be appropriately attributed.

## Getting the source

This repository uses git submodules, which must be checked out before building:

```sh
git clone --recurse-submodules <repo-url>
# or, on an existing clone
git submodule update --init --recursive
```

Most of the toolkit can be built and tested without Q.ANT hardware by enabling the `cpu-backend` Cargo feature, which is used throughout the examples below. Omit `-F cpu-backend` (and drop `--no-default-features` where noted) if you have Q.ANT hardware and drivers installed and want to test against them.

## Setting up a local development environment

Everything in this section works without Q.ANT hardware, and all three languages share a single conda environment. Pinned versions live in manifests rather than in these instructions, so CI and local environments cannot drift apart:

| Manifest | Owns |
| --- | --- |
| [`rust-toolchain.toml`](rust-toolchain.toml) | Rust toolchain channel and components |
| [`environment.yml`](environment.yml) | Python interpreter, pinned C/C++ test and Doxygen dependencies |
| [`pyproject.toml`](pyproject.toml) (`[dependency-groups]`) | Python tooling, test and doc dependencies |

### System packages

These come from your distribution and are not pinned by the repo (Debian/Ubuntu names):

```sh
sudo apt install build-essential curl git python3-dev \
                 libclang-dev clang clang-format patchelf lintian \
                 valgrind doxygen-latex graphviz
```

`libclang-dev` and `clang` are required by bindgen (see [`build.rs`](build.rs)), `python3-dev` by `cargo clippy --all-features`, `patchelf` and `lintian` by the Debian packaging, and `valgrind` by the C memory tests. CI pins `clang-format-19`; a different major version may disagree about formatting. `cmake` is *not* in this list — it is pinned in `environment.yml` instead.

### Development environment

Create it with conda ([miniforge](https://github.com/conda-forge/miniforge)):

```sh
conda env create -f environment.yml
conda activate qant-toolkit-dev
```

This is the only environment you need: it holds the Python interpreter as well as the pinned C/C++ test dependencies and Doxygen. The test CMake projects locate their headers through `$CONDA_PREFIX` (see [`tests_C/unit_tests/CMakeLists.txt`](tests_C/unit_tests/CMakeLists.txt)), so it also has to be active for the C tests and `gen_doxydoc.sh`. Doxygen is pinned because it is coupled to the `Doxyfile` and the vendored theme — see [`doc/doxygen/README.md`](doc/doxygen/README.md) before upgrading it.

### Rust toolchain

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

That is the only manual step. No `rustup install` or `rustup component add` is needed: `rust-toolchain.toml` pins the channel along with `rustfmt` and `clippy`, and rustup installs them on the first `cargo` invocation.

Two cargo tools are only needed when changing dependencies (see the [Rust](#rust) section below):

```sh
cargo install --locked cargo-about@0.9.0 cargo-machete
```

### Python packages

Python dependencies are pinned in `pyproject.toml`, not in `environment.yml`, so that CI installs exactly the same versions. With the conda environment active, pip installs them into it:

```sh
# the tests need torch, which is deliberately not part of the dev group
pip install --group torch --index-url https://download.pytorch.org/whl/cpu
pip install --group dev
maturin develop -F cpu-backend
```

Do not drop the `--index-url` on Linux or Windows: PyPI serves a torch that bundles CUDA, which is a 906 MB wheel plus ~2 GB of `nvidia-*` dependencies, against 175 MB for the CPU-only build. That is also why torch is not part of the `test` and `dev` groups — no plain `pip install --group …` can pull it in by accident. On macOS there is no CUDA build and plain `pip install --group torch` is correct.

`pip install --group` needs pip >= 25.1 for [PEP 735](https://peps.python.org/pep-0735/) dependency groups, which the conda environment provides.

**Without Q.ANT hardware, `-F cpu-backend` is not optional.** A plain `pip install .` (or `pip install -e .`) builds the default features, which link against `libqant_native_computing_driver` and therefore fail with a linker error on a machine that has no driver installed. `maturin develop` is where the feature flag can be passed, and it installs into the active environment. That is also why the development dependencies are PEP 735 dependency groups rather than extras: a group installs without building the extension module at all. To use pip for the build anyway:

```sh
MATURIN_PEP517_ARGS="--features cpu-backend" pip install -e .
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Rust core implementation |
| `src/c_bridge/` | C API bindings and Debian packaging (`packaging/create_deb.sh`) |
| `python/` | Python package sources (`qant_native_computing_toolkit/`) and pytest tests (`tests/`) |
| `tests_C/` | C unit tests (Catch2) and memory tests (Valgrind) |
| `doc/` | Sphinx (Python) and Doxygen (C) documentation sources |
| `external/` | Git submodules (e.g. dlpack) |

## Rust

```sh
cargo fmt --check
cargo clippy --all-features -- -D warnings
cargo test --benches --verbose --no-default-features -F cpu-backend
```

Notes:

- Clippy denies `unwrap()`, `expect()`, `panic!()`, and `mem::forget` outside of tests (see `clippy.toml`).
- CI also runs [`cargo-machete`](https://github.com/bnjbvr/cargo-machete) to flag unused dependencies.
- If you add or change a Rust dependency, its license must be in the allowlist in `dev_tools/about.toml`, and `THIRD_PARTY_LICENSES.md` must be regenerated:

  ```sh
  cargo clean && rm -f Cargo.lock
  ./dev_tools/generate_third_party_licenses.sh
  ```

  CI fails if the regenerated file differs from what's committed.

## Python

With the conda environment from the setup section active:

```sh
maturin develop -F cpu-backend

ruff check python/
ruff format --check python/
mypy ./python/
toml-sort --check pyproject.toml

pytest -m "not logging"
```

Building the docs:

```sh
sphinx-build -M html doc/python/source/ doc/python/build/ --fail-on-warning
```

## C / C++

```sh
cargo build --release --no-default-features -F cpu-backend
```

Formatting follows `.clang-format` (LLVM-based, Allman braces):

```sh
find . -path ./external -prune -o \( -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.h' \) \
  -exec clang-format --dry-run --Werror {} +
```

Tests live under `tests_C/` — see [`tests_C/README.md`](tests_C/README.md) for what they cover. They need the conda environment from the setup section to be active:

```sh
conda activate qant-toolkit-dev
cd tests_C/unit_tests && ./run_tests.sh
cd tests_C/memory_tests && ./run_tests.sh
```

Building the docs (also needs that environment, for the pinned Doxygen — see [`doc/doxygen/README.md`](doc/doxygen/README.md)):

```sh
conda activate qant-toolkit-dev
cd doc/doxygen/ && ./gen_doxydoc.sh
```

## CI

All of the above is enforced in CI (see `.github/workflows/`). Pull requests from forks run against the `cpu-backend` feature only, since the real-driver build requires access to Q.ANT hardware/drivers not available to external contributors.
