#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

from agentunlearn.config import ExperimentConfig
from agentunlearn.engine import ExperimentEngine
from agentunlearn.io import atomic_write_json, read_csv
from agentunlearn.models import LocalChatModel
from agentunlearn.schemas import RunSpec
from agentunlearn.training import apply_parameter_checkpoint


def model_paths(cfg: ExperimentConfig) -> dict[str, str]:
    rows = read_csv(cfg.paths.resolve(cfg.paths.models_manifest))
    return {r["model_id"]: r["model_path"] for r in rows}


def training_map(cfg: ExperimentConfig) -> dict[tuple[str, str, str], str]:
    p = cfg.paths.resolve(cfg.paths.training_manifest)
    if not p.exists():
        return {}
    rows = read_csv(p)
    return {(r["model_id"], r["unit_id"], r["parameter_method"]): r["training_id"] for r in rows}


def filtered_specs(args, cfg: ExperimentConfig):
    rows = read_csv(Path(args.manifest) if args.manifest else cfg.paths.resolve(cfg.paths.runs_manifest))
    specs = [RunSpec.from_mapping(r) for r in rows]
    if args.block:
        allowed = set(args.block)
        specs = [x for x in specs if x.block in allowed]
    if args.model:
        allowed = set(args.model)
        specs = [x for x in specs if x.model_id in allowed]
    if args.benchmark:
        allowed = set(args.benchmark)
        specs = [x for x in specs if x.benchmark in allowed]
    if args.run_id:
        allowed = set(args.run_id)
        specs = [x for x in specs if x.run_id in allowed]
    if args.max_runs is not None:
        specs = specs[: args.max_runs]
    return specs


def execute_specs(runtime: LocalChatModel, cfg: ExperimentConfig, specs, raw_dir: Path, error_dir: Path):
    engine = ExperimentEngine(cfg, runtime)
    for i, spec in enumerate(specs, 1):
        out = raw_dir / f"{spec.run_id}.json"
        err = error_dir / f"{spec.run_id}.json"
        if out.exists():
            print(f"[{i}/{len(specs)}] SKIP {spec.run_id} already complete", flush=True)
            continue
        print(
            f"[{i}/{len(specs)}] RUN {spec.run_id} {spec.model_id} {spec.benchmark}/{spec.unit_id} "
            f"{spec.topology} {spec.intervention}",
            flush=True,
        )
        try:
            result = engine.run_spec(spec)
            atomic_write_json(out, result.to_dict())
            if err.exists():
                err.unlink()
            print(f"  DONE {spec.run_id} {result.elapsed_sec:.1f}s", flush=True)
        except Exception as e:
            atomic_write_json(
                err,
                {
                    "run_id": spec.run_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"  FAIL {spec.run_id}: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    ap.add_argument("--manifest")
    ap.add_argument("--block", action="append")
    ap.add_argument("--model", action="append")
    ap.add_argument("--benchmark", action="append")
    ap.add_argument("--run-id", action="append")
    ap.add_argument("--max-runs", type=int)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    specs = filtered_specs(args, cfg)
    models = model_paths(cfg)
    train_lookup = training_map(cfg)

    outputs = cfg.paths.resolve(cfg.paths.outputs_dir)
    raw_dir = outputs / "raw"
    error_dir = outputs / "errors"
    raw_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    regular = defaultdict(list)
    parameter = defaultdict(list)
    for s in specs:
        if s.parameter_method and s.parameter_method != "none":
            parameter[(s.model_id, s.unit_id, s.parameter_method)].append(s)
        else:
            regular[s.model_id].append(s)

    # Normal inference: load each model once for all its conditions.
    for model_id, group in regular.items():
        if model_id not in models:
            raise KeyError(f"Model {model_id} missing from models_manifest")
        print(f"\n=== LOAD {model_id} ({len(group)} conditions) ===", flush=True)
        runtime = LocalChatModel(
            models[model_id],
            model_id,
            disable_qwen_thinking=cfg.generation.disable_qwen_thinking,
        ).load()
        execute_specs(runtime, cfg, group, raw_dir, error_dir)
        runtime.unload()

    # Parameter-level evaluation: reload base model per trained checkpoint to avoid cross-checkpoint contamination.
    for (model_id, unit_id, method), group in parameter.items():
        key = (model_id, unit_id, method)
        training_id = train_lookup.get(key)
        if not training_id:
            print(f"SKIP parameter group {key}: no training job mapping", flush=True)
            continue
        checkpoint = cfg.paths.resolve(cfg.paths.checkpoints_dir) / training_id
        if not (checkpoint / "metadata.json").exists():
            print(f"SKIP parameter group {key}: checkpoint not trained: {checkpoint}", flush=True)
            continue
        runtime = LocalChatModel(
            models[model_id],
            model_id,
            disable_qwen_thinking=cfg.generation.disable_qwen_thinking,
        ).load()
        apply_parameter_checkpoint(runtime, checkpoint, method)
        execute_specs(runtime, cfg, group, raw_dir, error_dir)
        runtime.unload()


if __name__ == "__main__":
    main()
