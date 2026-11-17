__version__ = "2.5.0"
__author__ = "Q.ANT GmbH"

import subprocess
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


def _get_dpkg_version(package: str) -> str:
    try:
        result = subprocess.run(
            ["dpkg", "-s", package],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
        return "not installed"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "dpkg not available"


# `maturin develop` does not remove backends from earlier builds. Python prefers the
# most specific suffix (e.g. `.cpython-312-x86_64-linux-gnu.so` over `.abi3.so`), so
# a stale build would otherwise be imported silently. A wheel only ever ships one.
_package_dir = Path(__file__).parent
_backends = sorted(
    p.name
    for p in _package_dir.glob("_backend.*")
    if any(p.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
)
if len(_backends) > 1:
    raise ImportError(
        f"Found multiple compiled backends {_backends} in {_package_dir}. "
        "Python would load the most specific one, which is likely a stale build. "
        "Delete them and rebuild, see CONTRIBUTING.md."
    )


# Attempt to import the wrapper module and provide detailed error information if it fails
try:
    from . import ai as ai
    from . import generic as generic
    from . import info as info
    from . import native as native
    from . import utils as utils

except ImportError as e:
    raise ImportError(
        f"""Failed to import the Q.ANT Toolkit. Please ensure that the NPU SDK is properly installed and configured. 
        Original error: {e}.
        Installed driver version: {_get_dpkg_version("qant_native_computing_driver")}.
        Installed toolkit version: {__version__}"""
    )
