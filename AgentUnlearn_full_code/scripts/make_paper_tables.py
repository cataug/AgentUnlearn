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
    args = ap.parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    mdir = cfg.paths.resolve(cfg.paths.outputs_dir) / "metrics"
    adir = cfg.paths.resolve(cfg.paths.outputs_dir) / "analysis"
    tdir = cfg.paths.resolve(cfg.paths.outputs_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    run_path = adir / "run_metrics_with_cd_ramp.csv"
    df = pd.read_csv(run_path if run_path.exists() else mdir / "run_metrics.csv")

    main = df[df.block == "main"].copy()
    table1 = (
        main.groupby(["model_id", "topology", "intervention"], as_index=False)
        .agg(
            DRR=("drr_target", "mean"),
            IRR=("irr_target", "mean"),
            Retention=("retention_target", "mean"),
            CD=("cd_target", "mean") if "cd_target" in main.columns else ("retention_target", lambda x: float("nan")),
        )
    )
    table1.to_csv(tdir / "table_main_results.csv", index=False)
    (tdir / "table_main_results.tex").write_text(
        table1.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=True),
        encoding="utf-8",
    )

    ab = df[df.block.str.startswith("ablation_", na=False)].copy()
    if not ab.empty:
        cols = [c for c in ["block", "n_agents", "chain_length", "target_agent_position", "debate_rounds", "peer_evidence", "comm_threshold", "irr_target", "retention_target", "cd_target"] if c in ab.columns]
        ab[cols].to_csv(tdir / "table_ablations_long.csv", index=False)

    param = df[df.block == "parameter_unlearning"].copy()
    if not param.empty:
        p = param.groupby(["model_id", "parameter_method", "topology", "intervention"], as_index=False).agg(
            DRR=("drr_target", "mean"), IRR=("irr_target", "mean"), Retention=("retention_target", "mean")
        )
        p.to_csv(tdir / "table_parameter_unlearning.csv", index=False)
        (tdir / "table_parameter_unlearning.tex").write_text(
            p.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
