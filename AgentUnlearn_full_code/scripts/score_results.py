#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from agentunlearn.config import ExperimentConfig
from agentunlearn.io import read_csv, write_csv
from agentunlearn.models import LocalChatModel
from agentunlearn.scoring import LLMJudge, score_run_dict
from agentunlearn.similarity import SimilarityEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    ap.add_argument("--judge-model-id")
    ap.add_argument("--max-runs", type=int)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    raw_dir = cfg.paths.resolve(cfg.paths.outputs_dir) / "raw"
    metrics_dir = cfg.paths.resolve(cfg.paths.outputs_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    sim = SimilarityEngine(cfg.similarity)
    judge = None
    judge_runtime = None
    if args.judge_model_id:
        models = {r["model_id"]: r["model_path"] for r in read_csv(cfg.paths.resolve(cfg.paths.models_manifest))}
        judge_runtime = LocalChatModel(
            models[args.judge_model_id], args.judge_model_id,
            disable_qwen_thinking=cfg.generation.disable_qwen_thinking,
        ).load()
        judge = LLMJudge(judge_runtime)

    probe_rows = []
    run_rows = []
    trace_rows = []
    files = sorted(raw_dir.glob("*.json"))
    if args.max_runs is not None:
        files = files[: args.max_runs]

    for i, p in enumerate(files, 1):
        with p.open("r", encoding="utf-8") as f:
            run = json.load(f)
        rows, summary, traces = score_run_dict(run, cfg, sim, judge)
        probe_rows.extend(rows)
        trace_rows.extend(traces)
        run_rows.append(summary)
        if i % 50 == 0 or i == len(files):
            print(f"scored {i}/{len(files)}", flush=True)

    write_csv(metrics_dir / "probe_scores.csv", probe_rows)
    write_csv(metrics_dir / "run_metrics.csv", run_rows)
    write_csv(metrics_dir / "trace_scores.csv", trace_rows)
    if judge_runtime is not None:
        judge_runtime.unload()


if __name__ == "__main__":
    main()
