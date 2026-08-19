__version__ = "2.3.0"
__author__ = "Q.ANT GmbH"

import subprocess


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
