"""Small, auditable fingerprints for run evidence and release gates."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from solarseg.data import sha256


def utc_now():
    return datetime.now(UTC).isoformat()


def source_hashes(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    files = sorted(root.glob("solarseg/*.py")) + [root / "requirements.txt"]
    return {str(p.relative_to(root)): sha256(p) for p in files}


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: version(p)
            for p in ["torch", "numpy", "pandas", "pillow", "scipy", "scikit-learn", "pycocotools"]
        },
    }
