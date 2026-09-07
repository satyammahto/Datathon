"""
AIDA Backend Package.
"""
import sys
from pathlib import Path

# Ensure the backend directory is in sys.path so 'app' is importable directly
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Ensure root directory is in sys.path so 'backend.app' is importable directly
_root_dir = _backend_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
