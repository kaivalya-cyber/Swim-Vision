# This file makes the project prefer repo-local Python packages and writable runtime caches.
"""Project-local Python startup customizations for SwimVision."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"

if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
