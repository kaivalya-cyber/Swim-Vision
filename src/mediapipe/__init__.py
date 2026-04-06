# This file redirects MediaPipe imports to the project-local Python 3.12 package when present.
"""Compatibility shim that prefers the project-local Python 3.12 MediaPipe package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_PACKAGE_DIR = PROJECT_ROOT / ".mp312" / "lib" / "python3.12" / "site-packages" / "mediapipe"
VENDORED_INIT = VENDORED_PACKAGE_DIR / "__init__.py"

if not VENDORED_INIT.exists():
    raise ModuleNotFoundError(
        "The project-local Python 3.12 MediaPipe package was not found at "
        f"'{VENDORED_PACKAGE_DIR}'."
    )

spec = importlib.util.spec_from_file_location(
    __name__,
    VENDORED_INIT,
    submodule_search_locations=[str(VENDORED_PACKAGE_DIR)],
)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to create an import spec for '{VENDORED_INIT}'.")

module = sys.modules[__name__]
module.__file__ = str(VENDORED_INIT)
module.__package__ = __name__
module.__path__ = [str(VENDORED_PACKAGE_DIR)]
module.__spec__ = spec
spec.loader.exec_module(module)
