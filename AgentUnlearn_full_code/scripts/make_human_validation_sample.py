#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd

from agentunlearn.config import ExperimentConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    mdir = cfg.paths.resolve(cfg.paths.outputs_dir) / "metrics"
    rawdir = cfg.paths.resolve(cfg.paths.outputs_dir) / "raw"
    scores = pd.read_csv(mdir / "probe_scores.csv")

    # Stratify by benchmark and probe kind where possible.
    scores = scores.reset_index(drop=True).copy()
    scores["_row_id"] = range(len(scores))
    groups = list(scores.groupby(["benchmark", "probe_kind"], dropna=False))
    quota = max(1, args.n // max(1, len(groups)))
    parts = []
    for _, g in groups:
        parts.append(g.sample(n=min(quota, len(g)), random_state=args.seed))
    sample = pd.concat(parts, ignore_index=True) if parts else scores.head(0).copy()
    chosen = set(sample["_row_id"].tolist())
    if len(sample) < args.n:
        remaining = scores[~scores["_row_id"].isin(chosen)]
        extra = remaining.sample(n=min(args.n - len(sample), len(remaining)), random_state=args.seed + 1)
        sample = pd.concat([sample, extra], ignore_index=True)
    sample = sample.head(args.n).copy()

    # Add raw response/question/reference fields.
    cache = {}
    prompts, refs, responses = [], [], []
    for _, r in sample.iterrows():
        rid = r.run_id
        if rid not in cache:
            import json
            cache[rid] = json.loads((rawdir / f"{rid}.json").read_text(encoding="utf-8"))
        run = cache[rid]
        match = next(
            p for p in run["probes"]
            if p["probe_id"] == r.probe_id and p["variant_id"] == r.variant_id
        )
        prompts.append(match["prompt"])
        refs.append(" || ".join(match["answers"]))
        responses.append(match["target_agent_output"])
    sample["question"] = prompts
    sample["reference_answers"] = refs
    sample["response"] = responses
    sample["human_label_a"] = ""  # 0=no reconstruction, 1=reconstruction
    sample["human_label_b"] = ""
    sample["human_notes_a"] = ""
    sample["human_notes_b"] = ""
    out = mdir / "human_validation_sample.csv"
    sample.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
