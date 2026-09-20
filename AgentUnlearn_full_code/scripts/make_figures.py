#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentunlearn.config import ExperimentConfig


TOPOLOGY_ORDER = ["independent", "sequential", "debate", "scratchpad", "hierarchical"]


def setup_fonts():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def fig_topology(run: pd.DataFrame, out: Path):
    d = run[(run.block == "main") & run.intervention.isin(["none", "target_context", "target_context_comm", "full_inference"])].copy()
    g = d.groupby(["topology", "intervention"], as_index=False).irr_target.mean()
    pivot = g.pivot(index="topology", columns="intervention", values="irr_target").reindex(TOPOLOGY_ORDER)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Indirect Reconstruction Rate (target agent)")
    ax.set_xlabel("Interaction topology")
    ax.set_title("Interaction topology and indirect reconstruction")
    ax.tick_params(axis="x", rotation=20)
    save(fig, out / "01_irr_by_topology.pdf")


def fig_trace_depth(trace: pd.DataFrame, out: Path):
    d = trace[
        trace.topology.isin(["sequential", "debate", "scratchpad"])
        & trace.intervention.isin(["target_context", "target_context_comm"])
        & (trace.probe_kind == "indirect")
    ].copy()
    if d.empty:
        return
    g = d.groupby(["topology", "intervention", "trace_index"], as_index=False).positive.mean()
    for topo in ["sequential", "debate", "scratchpad"]:
        x = g[g.topology == topo]
        if x.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for intervention, y in x.groupby("intervention"):
            ax.plot(y.trace_index, y.positive, marker="o", label=intervention)
        ax.set_xlabel("Interaction step")
        ax.set_ylabel("Reconstruction rate")
        ax.set_title(f"Knowledge reconstruction over interaction depth: {topo}")
        ax.legend()
        save(fig, out / f"02_depth_{topo}.pdf")


def fig_pareto(run: pd.DataFrame, out: Path):
    d = run[run.block == "ablation_comm_threshold"].copy()
    if d.empty or "cd_target" not in d.columns:
        return
    g = d.groupby("comm_threshold", as_index=False).agg(irr=("irr_target", "mean"), cd=("cd_target", "mean"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g.cd, g.irr, marker="o")
    for _, r in g.iterrows():
        ax.annotate(f"τ={r.comm_threshold:.2f}", (r.cd, r.irr), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Collateral Damage")
    ax.set_ylabel("Indirect Reconstruction Rate")
    ax.set_title("Communication-filter trade-off")
    save(fig, out / "03_irr_cd_pareto.pdf")


def fig_agent_scaling(run: pd.DataFrame, out: Path):
    d = run[run.block == "ablation_n_agents"].copy()
    if d.empty:
        return
    g = d.groupby(["n_agents", "topology", "intervention"], as_index=False).irr_target.mean()
    for topo in ["sequential", "debate", "scratchpad"]:
        x = g[g.topology == topo]
        if x.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for intervention, y in x.groupby("intervention"):
            ax.plot(y.n_agents, y.irr_target, marker="o", label=intervention)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Indirect Reconstruction Rate")
        ax.set_title(f"Reconstruction scaling with system size: {topo}")
        ax.legend()
        save(fig, out / f"04_agent_scaling_{topo}.pdf")


def fig_model_scale(run: pd.DataFrame, out: Path):
    d = run[run.block.isin(["main", "large_scale", "coder_scale"])].copy()
    if d.empty:
        return
    g = d.groupby(["model_id", "intervention"], as_index=False).irr_target.mean()
    pivot = g.pivot(index="model_id", columns="intervention", values="irr_target")
    keep = [x for x in ["none", "target_context", "target_context_comm"] if x in pivot.columns]
    if not keep:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot[keep].plot(kind="bar", ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Indirect Reconstruction Rate")
    ax.set_title("Model scale and specialization")
    ax.tick_params(axis="x", rotation=25)
    save(fig, out / "05_model_scale_specialization.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    setup_fonts()
    mdir = cfg.paths.resolve(cfg.paths.outputs_dir) / "metrics"
    adir = cfg.paths.resolve(cfg.paths.outputs_dir) / "analysis"
    fdir = cfg.paths.resolve(cfg.paths.outputs_dir) / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    run_path = adir / "run_metrics_with_cd_ramp.csv"
    run = pd.read_csv(run_path if run_path.exists() else mdir / "run_metrics.csv")
    trace = pd.read_csv(mdir / "trace_scores.csv") if (mdir / "trace_scores.csv").exists() else pd.DataFrame()
    fig_topology(run, fdir)
    if not trace.empty:
        fig_trace_depth(trace, fdir)
    fig_pareto(run, fdir)
    fig_agent_scaling(run, fdir)
    fig_model_scale(run, fdir)


if __name__ == "__main__":
    main()
