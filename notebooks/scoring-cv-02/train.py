"""Train independent baseline folds using the available Kaggle T4 devices.

This research script does not evaluate outer folds or submit predictions. Numerical
preprocessing packages are pinned; the image's CUDA-capable PyTorch build and the
complete installed environment are recorded. All training happens in fresh processes.
"""

import os
import subprocess
import sys
from pathlib import Path

SOURCE_COMMIT = "e9bb770a59650b364437a42e399a84e5421faac3"
REPOSITORY = "https://github.com/srivatsav-kannan/solar-seg.git"
ROOT = Path("/kaggle/working")
SOURCE = ROOT / "scoring-source"
OUTPUT = ROOT / "cv-02-baseline"
OUTPUT.mkdir(exist_ok=True)
subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(SOURCE)], check=True)
subprocess.run(["git", "checkout", SOURCE_COMMIT], cwd=SOURCE, check=True)

# Do not replace the CUDA PyTorch wheel with the CPU wheel used for public inference.
# No third-party numerical package is imported in this parent before installation.
used = {"numpy", "pandas", "pillow", "scipy", "scikit-learn", "pycocotools", "tqdm"}
pins = [
    line.strip()
    for line in (SOURCE / "requirements.txt").read_text().splitlines()
    if line.partition("==")[0] in used
]
subprocess.run([sys.executable, "-m", "pip", "install", *pins], check=True)
freeze = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"], check=True, text=True, capture_output=True
)
(OUTPUT / "environment-freeze.txt").write_text(freeze.stdout)

RUNNER = r"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from solarseg.data import CompetitionData, sha256
from solarseg.engine import prepare_cache
from solarseg.provenance import environment, utc_now

out = Path("/kaggle/working/cv-02-baseline")
data = CompetitionData("/kaggle/input")
index_path = Path("configs/scoring-cv-02/index.json")
index = json.loads(index_path.read_text())
if not torch.cuda.is_available():
    raise RuntimeError("A working CUDA GPU is required for this reserved run")
devices = min(torch.cuda.device_count(), 2)
sanity = torch.from_numpy(np.ones((8, 8), np.float32)).to("cuda").sum().item()
assert sanity == 64
record = {
    "created_at": utc_now(), "source_commit": "e9bb770a59650b364437a42e399a84e5421faac3",
    "environment": environment(), "gpu_count": devices,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(devices)],
    "cuda_version": torch.version.cuda, "index_sha256": sha256(index_path),
    "training_only": True, "outer_evaluation_performed": False,
    "protocol": "Three independent baseline fits. Each fit reads optimization observations only; original calibration and its outer holdout are excluded.",
}
(out / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)

# Prepare the union of optimization images before concurrent fits, avoiding cache
# write races. Each worker loads only its own manifest's train rows.
frames = []
for fold in index["outer_folds"]:
    item = index["records"][str(fold)]
    assert sha256(item["path"]) == item["sha256"]
    frame = pd.read_csv(item["path"])
    frames.append(frame.loc[frame.role == "train"])
pool = pd.concat(frames).drop_duplicates("stem")
cache = out / "cache-1024"
prepare_cache(data, pool, cache, 1024)

pending = list(index["outer_folds"])
running = {}
results = []
started = time.monotonic()
last_progress = started
try:
    while pending or running:
        for device in range(devices):
            if device in running or not pending:
                continue
            fold = pending.pop(0)
            run = out / f"baseline-fold-{fold}"
            log = (out / f"baseline-fold-{fold}.log").open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            command = [sys.executable, "-u", "-m", "research.train", "--data", "/kaggle/input",
                       "--manifest", index["records"][str(fold)]["path"], "--run", str(run),
                       "--size", "1024", "--crop", "384", "--batch-size", "6", "--width", "24",
                       "--depth", "3", "--steps", "6000", "--seed", "2026", "--device", "cuda"]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[device] = (fold, process, log, run)
            print("START", fold, "GPU", device, flush=True)
        for device, (fold, process, log, run) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            result = {"fold": fold, "exit_code": code, "completed": code == 0 and (run / "model.pt").exists()}
            if result["completed"]:
                result["checkpoint_sha256"] = sha256(run / "model.pt")
                result["training_config"] = json.loads((run / "config.json").read_text())
            results.append(result)
            del running[device]
            print("FINISHED", fold, "exit", code, flush=True)
            (out / "fit-results.json").write_text(json.dumps(results, indent=2) + "\n")
            if code:
                raise RuntimeError(f"Fold {fold} failed; inspect its saved log")
        if time.monotonic() - started > 6300:
            raise TimeoutError("Stopping this training reservation before the kernel's time limit")
        if time.monotonic() - last_progress >= 60:
            for device, (fold, process, log, run) in running.items():
                history = run / "history.json"
                if history.exists():
                    try:
                        print("PROGRESS", fold, json.loads(history.read_text())[-1], flush=True)
                    except json.JSONDecodeError:
                        pass
            last_progress = time.monotonic()
        time.sleep(5)
finally:
    for fold, process, log, run in running.values():
        process.terminate()
        process.wait(timeout=30)
        log.close()

assert len(results) == 3 and all(r["completed"] for r in results)
shutil.rmtree(cache)
checks = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in out.rglob("*") if p.is_file() and p.name != "checksums.json"}
(out / "checksums.json").write_text(json.dumps(checks, indent=2) + "\n")
print("ALL THREE BASELINE FITS COMPLETE", flush=True)
"""

runner = ROOT / "run_cv_baselines.py"
runner.write_text(RUNNER)
subprocess.run(
    [sys.executable, str(runner)],
    cwd=SOURCE,
    check=True,
    env={**os.environ, "PYTHONPATH": str(SOURCE)},
)
