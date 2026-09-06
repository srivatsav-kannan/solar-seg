"""Run meaningful contracts and save machine-readable evidence for the gates."""

import json
import subprocess
import sys
from pathlib import Path

from solarseg.data import sha256
from solarseg.provenance import source_hashes, utc_now

commands = [
    [sys.executable, "-m", "pytest", "-q", "-rA"],
    [sys.executable, "-m", "ruff", "check", "solarseg", "scripts", "tests"],
]
results = []
for command in commands:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    results.append(
        {
            "command": command[2:],
            "returncode": result.returncode,
            "output": result.stdout + result.stderr,
        }
    )
    print(result.stdout, result.stderr)
official = Path("references/official/self-evaluation-notebook.ipynb")
record = {
    "created_at": utc_now(),
    "passed": all(r["returncode"] == 0 for r in results),
    "organizer_parity_passed": "PASSED tests/test_contracts.py::test_parity_with_downloaded_organizer_evaluator"
    in results[0]["output"],
    "organizer_notebook_sha256": sha256(official) if official.exists() else None,
    "source_hashes": source_hashes(),
    "checks": results,
}
Path("reports/quality-checks.json").write_text(json.dumps(record, indent=2))
if not record["passed"]:
    raise SystemExit(1)
