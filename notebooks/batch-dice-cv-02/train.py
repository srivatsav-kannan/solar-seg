"""Test native refinement and batch-Dice fine-tuning on the available Kaggle T4 devices.

This research script does not evaluate outer folds or submit predictions. Numerical
preprocessing packages are pinned; the image's CUDA-capable PyTorch build and the
complete installed environment are recorded. All training happens in fresh processes.
"""

import os
import subprocess
import sys
from pathlib import Path

SOURCE_COMMIT = "7854fc9c5dc4858d8561865dfad59a6d63a2c5aa"
REPOSITORY = "https://github.com/srivatsav-kannan/solar-seg.git"
ROOT = Path("/kaggle/working")
SOURCE = ROOT / "batch-cv-source"
OUTPUT = ROOT / "batch-dice-cv-02"
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

RUNNER = '"""Bounded CUDA warm-start screens; training caches never enter published outputs."""\n\nimport json\nimport os\nimport subprocess\nimport sys\nimport time\nimport urllib.request\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport torch\n\nfrom solarseg.data import CompetitionData, sha256\nfrom solarseg.engine import prepare_cache\nfrom solarseg.provenance import environment, utc_now\n\nout = Path("/kaggle/working/batch-dice-cv-02")\nplan_path = Path("configs/batch-dice-cv-02.json")\nplan = json.loads(plan_path.read_text())\ndata = CompetitionData("/kaggle/input")\nif not torch.cuda.is_available():\n    raise RuntimeError("This bounded training job requires a working CUDA device")\nassert torch.from_numpy(np.ones((8, 8), np.float32)).cuda().sum().item() == 64\ndevices = min(torch.cuda.device_count(), 2)\nrecord = {\n    "created_at": utc_now(),\n    "environment": environment(),\n    "gpu_count": devices,\n    "gpu_names": [torch.cuda.get_device_name(i) for i in range(devices)],\n    "plan_sha256": sha256(plan_path),\n    "training_only": True,\n    "holdout_or_test_evaluation": False,\n}\n(out / "environment.json").write_text(json.dumps(record, indent=2) + "\\n")\n(out / "plan.json").write_text(json.dumps(plan, indent=2) + "\\n")\nprint(json.dumps(record), flush=True)\ncache_root = Path("/tmp/solarseg-cv-warm-cache")\nmodels = []\nframes = []\nfor fold in plan["folds"]:\n    if fold["device"] != "cuda":\n        continue\n    initial = Path(f"/tmp/solarseg-parent-fold-{fold[\'fold\']}.pt")\n    urllib.request.urlretrieve(\n        f"https://github.com/srivatsav-kannan/solar-seg/releases/download/cv-parents-v0.1/fold-{fold[\'fold\']}-model.pt",\n        initial,\n    )\n    assert sha256(initial) == fold["parent_checkpoint_sha256"]\n    assert sha256(fold["manifest"]) == fold["manifest_sha256"]\n    frame = pd.read_csv(fold["manifest"])\n    frames.append(frame.loc[frame.role == "train"])\n    models.append({**plan["recipe"], **fold,\n                   "run": f"batch-dice-fold-{fold[\'fold\']}", "initial": str(initial)})\nassert [cfg["fold"] for cfg in models] == [2, 3]\nassert devices == 2, "Require two T4s to stay within the reserved runtime"\npool = pd.concat(frames).drop_duplicates("stem")\nprepare_cache(data, pool, cache_root / "cache-1024", 1024)\npending = list(models)\nrunning = {}\nresults = []\nstarted = last_progress = time.monotonic()\ntry:\n    while pending or running:\n        for device in range(devices):\n            if device in running or not pending:\n                continue\n            cfg = pending.pop(0)\n            name = cfg["run"]\n            run = out / name\n            log = (out / f"{name}.log").open("w")\n            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}\n            command = [\n                sys.executable,\n                "-u",\n                "-m",\n                "research.train",\n                "--data",\n                "/kaggle/input",\n                "--manifest",\n                cfg["manifest"],\n                "--run",\n                str(run),\n                "--initialize",\n                cfg["initial"],\n                "--cache-root",\n                str(cache_root),\n                "--loss-mode",\n                cfg["loss_mode"],\n                "--device",\n                "cuda",\n            ]\n            for key in [\n                "size",\n                "crop",\n                "batch_size",\n                "width",\n                "depth",\n                "steps",\n                "seed",\n                "learning_rate",\n            ]:\n                command.extend(["--" + key.replace("_", "-"), str(cfg[key])])\n            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)\n            running[device] = (name, process, log, run)\n            print("START", name, "GPU", device, flush=True)\n        for device, (name, process, log, run) in list(running.items()):\n            code = process.poll()\n            if code is None:\n                continue\n            log.close()\n            result = {\n                "run": name,\n                "exit_code": code,\n                "completed": code == 0 and (run / "model.pt").exists(),\n            }\n            if result["completed"]:\n                result["checkpoint_sha256"] = sha256(run / "model.pt")\n                result["config"] = json.loads((run / "config.json").read_text())\n            results.append(result)\n            del running[device]\n            (out / "fit-results.json").write_text(json.dumps(results, indent=2) + "\\n")\n            print("FINISHED", name, "exit", code, flush=True)\n            if code:\n                raise RuntimeError(f"Training failed for {name}; inspect the saved log")\n        if time.monotonic() - started > 1200:\n            raise TimeoutError("Bounded warm-start training deadline reached")\n        if time.monotonic() - last_progress >= 60:\n            for name, process, log, run in running.values():\n                history = run / "history.json"\n                if history.exists():\n                    try:\n                        print("PROGRESS", name, json.loads(history.read_text())[-1], flush=True)\n                    except json.JSONDecodeError:\n                        pass\n            last_progress = time.monotonic()\n        time.sleep(5)\nfinally:\n    for name, process, log, run in running.values():\n        process.terminate()\n        process.wait(timeout=30)\n        log.close()\n\nassert len(results) == 2 and all(r["completed"] for r in results)\nchecksums = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}\n(out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\\n")\nprint("BOTH CV FINE-TUNING FITS COMPLETE", flush=True)\n'

runner = ROOT / "run_batch_cv.py"
runner.write_text(RUNNER)
subprocess.run(
    [sys.executable, str(runner)],
    cwd=SOURCE,
    check=True,
    env={**os.environ, "PYTHONPATH": str(SOURCE)},
)
