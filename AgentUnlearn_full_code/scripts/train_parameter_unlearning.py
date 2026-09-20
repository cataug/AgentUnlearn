#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import traceback
from pathlib import Path

from agentunlearn.config import ExperimentConfig
from agentunlearn.io import atomic_write_json, read_csv
from agentunlearn.training import train_parameter_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    ap.add_argument("--training-id", action="append")
    ap.add_argument("--model", action="append")
    ap.add_argument("--method", action="append")
    ap.add_argument("--max-jobs", type=int)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    models = {r["model_id"]: r["model_path"] for r in read_csv(cfg.paths.resolve(cfg.paths.models_manifest))}
    rows = read_csv(cfg.paths.resolve(cfg.paths.training_manifest))

    if args.training_id:
        allowed = set(args.training_id)
        rows = [r for r in rows if r["training_id"] in allowed]
    if args.model:
        allowed = set(args.model)
        rows = [r for r in rows if r["model_id"] in allowed]
    if args.method:
        allowed = set(args.method)
        rows = [r for r in rows if r["parameter_method"] in allowed]
    if args.max_jobs is not None:
        rows = rows[: args.max_jobs]

    base = cfg.paths.resolve(cfg.paths.checkpoints_dir)
    base.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(rows, 1):
        out = base / r["training_id"]
        if (out / "metadata.json").exists():
            print(f"[{i}/{len(rows)}] SKIP {r['training_id']} already trained", flush=True)
            continue
        print(f"[{i}/{len(rows)}] TRAIN {r['training_id']} {r['model_id']} {r['unit_id']} {r['parameter_method']}", flush=True)
        try:
            train_parameter_method(
                cfg=cfg,
                model_id=r["model_id"],
                model_path=models[r["model_id"]],
                benchmark=r["benchmark"],
                unit_id=r["unit_id"],
                target_id=r.get("target_id", ""),
                target_name=r.get("target_name", ""),
                method=r["parameter_method"],
                output_dir=out,
            )
            print(f"  DONE {r['training_id']}", flush=True)
        except Exception as e:
            out.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                out / "error.json",
                {"error_type": type(e).__name__, "error": str(e), "traceback": traceback.format_exc()},
            )
            print(f"  FAIL {r['training_id']}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
