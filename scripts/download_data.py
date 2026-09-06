"""Download the official competition bundle and extract it safely."""

import argparse
import subprocess
import zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--destination", default="data/raw")
a = p.parse_args()
destination = Path(a.destination)
destination.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["kaggle", "competitions", "download", "filament-segmentation-2026", "-p", str(destination)],
    check=True,
)
with zipfile.ZipFile(destination / "filament-segmentation-2026.zip") as z:
    for member in z.namelist():
        if Path(member).is_absolute() or ".." in Path(member).parts:
            raise ValueError("Unsafe archive path")
    if z.testzip() is not None:
        raise ValueError("Corrupt archive")
    z.extractall(destination)
