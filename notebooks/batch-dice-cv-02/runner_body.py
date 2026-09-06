"""Bounded CUDA warm-start screens; training caches never enter published outputs."""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from solarseg.data import CompetitionData, sha256
from solarseg.engine import prepare_cache
from solarseg.provenance import environment, utc_now

out = Path("/kaggle/working/batch-dice-cv-02")
plan_path = Path("configs/batch-dice-cv-02.json")
plan = json.loads(plan_path.read_text())
data = CompetitionData("/kaggle/input")
if not torch.cuda.is_available():
    raise RuntimeError("This bounded training job requires a working CUDA device")
assert torch.from_numpy(np.ones((8, 8), np.float32)).cuda().sum().item() == 64
devices = min(torch.cuda.device_count(), 2)
record = {
    "created_at": utc_now(),
    "environment": environment(),
    "gpu_count": devices,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(devices)],
    "plan_sha256": sha256(plan_path),
    "training_only": True,
    "holdout_or_test_evaluation": False,
}
(out / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
(out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(json.dumps(record), flush=True)
cache_root = Path("/tmp/solarseg-cv-warm-cache")
models = []
frames = []
for fold in plan["folds"]:
    if fold["device"] != "cuda":
        continue
    initial = Path(f"/tmp/solarseg-parent-fold-{fold['fold']}.pt")
    urllib.request.urlretrieve(
        f"https://github.com/srivatsav-kannan/solar-seg/releases/download/cv-parents-v0.1/fold-{fold['fold']}-model.pt",
        initial,
    )
    assert sha256(initial) == fold["parent_checkpoint_sha256"]
    assert sha256(fold["manifest"]) == fold["manifest_sha256"]
    frame = pd.read_csv(fold["manifest"])
    frames.append(frame.loc[frame.role == "train"])
    models.append(
        {
            **plan["recipe"],
            **fold,
            "run": f"batch-dice-fold-{fold['fold']}",
            "initial": str(initial),
        }
    )
assert [cfg["fold"] for cfg in models] == [2, 3]
assert devices == 2, "Require two T4s to stay within the reserved runtime"
pool = pd.concat(frames).drop_duplicates("stem")
prepare_cache(data, pool, cache_root / "cache-1024", 1024)
pending = list(models)
running = {}
results = []
started = last_progress = time.monotonic()
try:
    while pending or running:
        for device in range(devices):
            if device in running or not pending:
                continue
            cfg = pending.pop(0)
            name = cfg["run"]
            run = out / name
            log = (out / f"{name}.log").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}
            command = [
                sys.executable,
                "-u",
                "-m",
                "research.train",
                "--data",
                "/kaggle/input",
                "--manifest",
                cfg["manifest"],
                "--run",
                str(run),
                "--initialize",
                cfg["initial"],
                "--cache-root",
                str(cache_root),
                "--loss-mode",
                cfg["loss_mode"],
                "--device",
                "cuda",
            ]
            for key in [
                "size",
                "crop",
                "batch_size",
                "width",
                "depth",
                "steps",
                "seed",
                "learning_rate",
            ]:
                command.extend(["--" + key.replace("_", "-"), str(cfg[key])])
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[device] = (name, process, log, run)
            print("START", name, "GPU", device, flush=True)
        for device, (name, process, log, run) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            result = {
                "run": name,
                "exit_code": code,
                "completed": code == 0 and (run / "model.pt").exists(),
            }
            if result["completed"]:
                result["checkpoint_sha256"] = sha256(run / "model.pt")
                result["config"] = json.loads((run / "config.json").read_text())
            results.append(result)
            del running[device]
            (out / "fit-results.json").write_text(json.dumps(results, indent=2) + "\n")
            print("FINISHED", name, "exit", code, flush=True)
            if code:
                raise RuntimeError(f"Training failed for {name}; inspect the saved log")
        if time.monotonic() - started > 1200:
            raise TimeoutError("Bounded warm-start training deadline reached")
        if time.monotonic() - last_progress >= 60:
            for name, process, log, run in running.values():
                history = run / "history.json"
                if history.exists():
                    try:
                        print("PROGRESS", name, json.loads(history.read_text())[-1], flush=True)
                    except json.JSONDecodeError:
                        pass
            last_progress = time.monotonic()
        time.sleep(5)
finally:
    for name, process, log, run in running.values():
        process.terminate()
        process.wait(timeout=30)
        log.close()

assert len(results) == 2 and all(r["completed"] for r in results)
checksums = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}
(out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
print("BOTH CV FINE-TUNING FITS COMPLETE", flush=True)
