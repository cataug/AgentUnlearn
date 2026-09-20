#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from agentunlearn.config import ExperimentConfig
from agentunlearn.io import read_csv, write_csv
from agentunlearn.models import LocalChatModel
from agentunlearn.training import apply_parameter_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    ap.add_argument("--block", action="append")
    ap.add_argument("--model", action="append")
    ap.add_argument("--max-runs", type=int)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    models = {r["model_id"]: r["model_path"] for r in read_csv(cfg.paths.resolve(cfg.paths.models_manifest))}
    train_rows = read_csv(cfg.paths.resolve(cfg.paths.training_manifest))
    train_map = {(r["model_id"], r["unit_id"], r["parameter_method"]): r["training_id"] for r in train_rows}

    raw_dir = cfg.paths.resolve(cfg.paths.outputs_dir) / "raw"
    files = sorted(raw_dir.glob("*.json"))
    grouped = defaultdict(list)
    count = 0
    for p in files:
        run = json.loads(p.read_text(encoding="utf-8"))
        s = run["spec"]
        if args.block and s["block"] not in set(args.block):
            continue
        if args.model and s["model_id"] not in set(args.model):
            continue
        key = (s["model_id"], s["unit_id"], s.get("parameter_method", "none"))
        grouped[key].append(run)
        count += 1
        if args.max_runs is not None and count >= args.max_runs:
            break

    rows = []
    for (model_id, unit_id, method), runs in grouped.items():
        runtime = LocalChatModel(
            models[model_id], model_id,
            disable_qwen_thinking=cfg.generation.disable_qwen_thinking,
        ).load()
        if method != "none":
            tid = train_map[(model_id, unit_id, method)]
            apply_parameter_checkpoint(runtime, cfg.paths.resolve(cfg.paths.checkpoints_dir) / tid, method)

        for run in runs:
            s = run["spec"]
            for p in run.get("probes", []):
                target_traces = [t for t in p.get("traces", []) if t.get("is_target")]
                if not target_traces or not p.get("answers"):
                    continue
                messages = target_traces[-1]["prompt_messages"]
                with runtime.agent_parameter_context(True):
                    vals = [runtime.answer_log_likelihood(messages, a) for a in p["answers"] if a]
                rows.append({
                    "run_id": s["run_id"],
                    "model_id": model_id,
                    "benchmark": s["benchmark"],
                    "unit_id": unit_id,
                    "topology": s["topology"],
                    "intervention": s["intervention"],
                    "parameter_method": method,
                    "probe_id": p["probe_id"],
                    "probe_kind": p["probe_kind"],
                    "variant_id": p["variant_id"],
                    "answer_loglikelihood": max(vals) if vals else float("nan"),
                })
        runtime.unload()

    metrics_dir = cfg.paths.resolve(cfg.paths.outputs_dir) / "metrics"
    write_csv(metrics_dir / "loglikelihood_scores.csv", rows)


if __name__ == "__main__":
    main()
