"""Read an existing upload's server result without uploading anything."""

import argparse
import json
from pathlib import Path

from kaggle import api

from solarseg.provenance import utc_now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    path = Path(args.run) / "kaggle-submission.json"
    record = json.loads(path.read_text())
    ref = record.get("response", {}).get("ref")
    if ref is None:
        raise RuntimeError(
            "Upload outcome is uncertain. Inspect account submissions; do not retry."
        )
    rows = api.competition_submissions(record["competition"])
    matches = [json.loads(str(row)) for row in rows if row is not None and row.ref == ref]
    if len(matches) != 1:
        raise RuntimeError(f"Submission {ref} was not found in the current page. Do not re-upload.")
    server = matches[0]
    record.update(verified_at=utc_now(), server=server, state=server["status"])
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {k: server.get(k) for k in ["ref", "status", "publicScore", "errorDescription"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
