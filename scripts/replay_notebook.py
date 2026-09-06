"""Execute canonical inference in a fresh kernel and compare the exact local CSV."""

import argparse
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from solarseg.data import sha256
from solarseg.provenance import source_hashes, utc_now

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument(
    "--record-only",
    action="store_true",
    help="Finalize an already executed replay after its independent reference CSV is ready",
)
a = p.parse_args()
notebook_path = root / "notebooks/canonical.ipynb"
if not a.record_only:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name=os.environ.get("SOLARSEG_KERNEL", "solarseg"),
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    nbformat.write(notebook, root / "reports/canonical-executed.ipynb")
else:
    executed = nbformat.read(root / "reports/canonical-executed.ipynb", as_version=4)
    canonical = nbformat.read(notebook_path, as_version=4)
    if [c.source for c in executed.cells] != [c.source for c in canonical.cells]:
        raise ValueError("Executed notebook does not match current canonical source")
    for cell in executed.cells:
        if cell.cell_type == "code" and (
            cell.execution_count is None or any(o.output_type == "error" for o in cell.outputs)
        ):
            raise ValueError("Notebook execution is incomplete or contains an error")
selected = json.loads((root / "configs/selected.json").read_text())
proof = json.loads((root / "artifacts/notebook-run/notebook-proof.json").read_text())
expected = sha256(root / selected["run"] / "submission.csv")
record = {
    "created_at": utc_now(),
    "passed": proof["completed"] and proof["validation"]["sha256"] == expected,
    "submission_sha256": proof["validation"]["sha256"],
    "expected_submission_sha256": expected,
    "notebook_sha256": sha256(notebook_path),
    "source_hashes": source_hashes(),
    "proof": proof,
    "scope": "Fresh kernel, full-input audit and CPU inference replay; training was not rerun",
}
out = root / "artifacts/replay"
out.mkdir(parents=True, exist_ok=True)
(out / "record.json").write_text(json.dumps(record, indent=2))
if not record["passed"]:
    raise RuntimeError("Notebook replay differs from the selected local CPU submission")
print(json.dumps({k: v for k, v in record.items() if k != "proof"}, indent=2))
