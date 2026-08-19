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
  cargo install cargo-about --version 0.9.0
  ./dev_tools/generate_third_party_licenses.sh
  ```

  CI fails if the regenerated file differs from what's committed.

## Python

Requires the Rust toolchain and [maturin](https://www.maturin.rs/) (`pip install maturin`).

```sh
maturin develop -F cpu-backend

ruff check python/
ruff format --check python/
mypy ./python/
toml-sort --check pyproject.toml

pytest -m "not logging"
```

Building the docs (`pip install .[doc]`):

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

Tests live under `tests_C/` and require Catch2, xtensor, xtensor-blas (installable via conda) and Valgrind — see [`tests_C/README.md`](tests_C/README.md) for details:

```sh
cd tests_C/unit_tests && ./run_tests.sh
cd tests_C/memory_tests && ./run_tests.sh
```

Building the docs:

```sh
cd doc/doxygen/ && ./gen_doxydoc.sh
```

## CI

All of the above is enforced in CI (see `.github/workflows/`). Pull requests from forks run against the `cpu-backend` feature only, since the real-driver build requires access to Q.ANT hardware/drivers not available to external contributors.
