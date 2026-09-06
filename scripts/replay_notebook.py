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
p.add_argument("--notebook", default="notebooks/canonical.ipynb")
p.add_argument("--selection", default="configs/selected.json")
p.add_argument("--work", default="artifacts/notebook-run")
p.add_argument("--executed", default="reports/canonical-executed.ipynb")
p.add_argument("--record", default="artifacts/replay/record.json")
a = p.parse_args()
notebook_path = root / a.notebook
work = root / a.work
executed_path = root / a.executed
executed_path.parent.mkdir(parents=True, exist_ok=True)
start_path = executed_path.with_suffix(".start.json")
notebook_sha = sha256(notebook_path)
cache_images_before = len(list((work / "probabilities-test").glob("*.npy")))
if not a.record_only:
    start_path.write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "notebook_sha256": notebook_sha,
                "cache_images_before": cache_images_before,
            },
            indent=2,
        )
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name=os.environ.get("SOLARSEG_KERNEL", "solarseg"),
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    nbformat.write(notebook, executed_path)
    if sha256(notebook_path) != notebook_sha:
        raise RuntimeError(
            "Canonical notebook changed during execution; evidence was not finalized"
        )
else:
    executed = nbformat.read(executed_path, as_version=4)
    canonical = nbformat.read(notebook_path, as_version=4)
    if [c.source for c in executed.cells] != [c.source for c in canonical.cells]:
        raise ValueError("Executed notebook does not match current canonical source")
    for cell in executed.cells:
        if cell.cell_type == "code" and (
            cell.execution_count is None or any(o.output_type == "error" for o in cell.outputs)
        ):
            raise ValueError("Notebook execution is incomplete or contains an error")
    if start_path.exists():
        start = json.loads(start_path.read_text())
        assert start["notebook_sha256"] == notebook_sha
        cache_images_before = start["cache_images_before"]
    else:
        cache_images_before = None
selected = json.loads((root / a.selection).read_text())
proof = json.loads((work / "notebook-proof.json").read_text())
expected = sha256(root / selected["run"] / "submission.csv")
reference_validation = json.loads(
    (root / selected["run"] / "submission.validation.json").read_text()
)
assert reference_validation["valid"] and reference_validation["sha256"] == expected
record = {
    "created_at": utc_now(),
    "passed": proof["completed"] and proof["validation"]["sha256"] == expected,
    "submission_sha256": proof["validation"]["sha256"],
    "expected_submission_sha256": expected,
    "notebook_sha256": notebook_sha,
    "source_hashes": source_hashes(),
    "proof": proof,
    "cache_images_before": cache_images_before,
    "selection_sha256": sha256(root / a.selection),
    "scope": (
        "Fresh kernel, full-input audit and CPU prediction/serialization replay; "
        "fingerprinted caches may be reused when cache_images_before is nonzero. Training was not rerun."
    ),
}
out = root / a.record
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2))
if not record["passed"]:
    raise RuntimeError("Notebook replay differs from the selected local CPU submission")
print(json.dumps({k: v for k, v in record.items() if k != "proof"}, indent=2))
