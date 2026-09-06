"""Upload one gated CSV through the Kaggle SDK; retain uncertain attempts."""

import argparse
import json
from pathlib import Path

from kaggle import api

from solarseg.gates import build_gates
from solarseg.provenance import utc_now

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--data", default="data/raw")
p.add_argument("--check-only", action="store_true")
a = p.parse_args()
run = Path(a.run)
competition = "filament-segmentation-2026"
gate = build_gates(run, data_root=a.data)
limits = json.loads(str(api.competition_get_submission_limits(competition)))
if limits.get("numAllowedNow", 0) < 1:
    raise RuntimeError(f"No submissions available: {limits}")
print(json.dumps({"gates": gate, "limits": limits}, indent=2))
if not a.check_only:
    path = run / "kaggle-submission.json"
    # An interrupted request may already have uploaded. Investigate the account
    # status before deliberately changing this record; never automatically retry.
    with path.open("x") as f:
        record = {
            "created_at": utc_now(),
            "state": "intent_recorded",
            "competition": competition,
            "submission_sha256": gate["submission_sha256"],
            "description": f"solar-seg baseline | grouped holdout PQ {gate['holdout_pq']:.4f} | {gate['submission_sha256'][:12]}",
            "limits_before": limits,
        }
        json.dump(record, f, indent=2)
    response = api.competition_submit(
        str(run / "submission.csv"), record["description"], competition
    )
    record["response"] = json.loads(str(response))
    record["state"] = "response_received_check_server_status"
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
