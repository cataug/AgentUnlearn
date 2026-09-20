#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agentunlearn.config import ExperimentConfig
from agentunlearn.io import write_csv
from agentunlearn.stats import mcnemar_exact, paired_bootstrap_difference, wilson_interval


def add_cd_and_ramp(run: pd.DataFrame) -> pd.DataFrame:
    df = run.copy()
    keys = ["model_id", "benchmark", "unit_id", "topology"]
    base = (
        df[(df["intervention"] == "none") & (df["parameter_method"] == "none")]
        [keys + ["retention_target", "retention_system"]]
        .rename(columns={"retention_target": "baseline_retention_target", "retention_system": "baseline_retention_system"})
    )
    df = df.merge(base, on=keys, how="left")
    for scope in ["target", "system"]:
        r = df[f"retention_{scope}"]
        b = df[f"baseline_retention_{scope}"]
        cd = 1 - (r / b)
        cd[(~np.isfinite(cd)) | (b <= 0)] = np.nan
        df[f"cd_{scope}"] = cd.clip(lower=0)

    # Reconstruction amplification against Independent with otherwise identical settings.
    ramp_keys = ["model_id", "benchmark", "unit_id", "intervention", "parameter_method"]
    indep = (
        df[df["topology"] == "independent"]
        [ramp_keys + ["irr_target", "irr_system"]]
        .rename(columns={"irr_target": "irr_target_independent", "irr_system": "irr_system_independent"})
    )
    df = df.merge(indep, on=ramp_keys, how="left")
    eps = 1e-6
    df["ramp_target"] = (df["irr_target"] + eps) / (df["irr_target_independent"] + eps)
    df["ramp_system"] = (df["irr_system"] + eps) / (df["irr_system_independent"] + eps)
    return df


def aggregate_probe_rates(probes: pd.DataFrame) -> pd.DataFrame:
    group = ["model_id", "benchmark", "topology", "intervention", "parameter_method", "probe_kind"]
    rows = []
    for keys, g in probes.groupby(group, dropna=False):
        row = dict(zip(group, keys))
        for scope in ["target", "system"]:
            vals = g[f"{scope}_positive"].dropna().astype(int)
            k, n = int(vals.sum()), int(len(vals))
            lo, hi = wilson_interval(k, n)
            row[f"{scope}_rate"] = k / n if n else np.nan
            row[f"{scope}_n"] = n
            row[f"{scope}_ci_low"] = lo
            row[f"{scope}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(probes: pd.DataFrame) -> pd.DataFrame:
    # Primary paired comparison requested by the design: TargetCtx vs TargetCtx+Comm.
    pairs = [
        ("target_context", "target_context_comm"),
        ("none", "target_context"),
        ("none", "full_inference"),
    ]
    idx = ["model_id", "benchmark", "unit_id", "topology", "probe_id", "probe_kind", "variant_id", "temperature", "seed"]
    out = []
    for a_name, b_name in pairs:
        a = probes[probes.intervention == a_name][idx + ["target_positive"]].rename(columns={"target_positive": "a"})
        b = probes[probes.intervention == b_name][idx + ["target_positive"]].rename(columns={"target_positive": "b"})
        m = a.merge(b, on=idx, how="inner")
        if m.empty:
            continue
        for (model, bench, topology, kind), g in m.groupby(["model_id", "benchmark", "topology", "probe_kind"]):
            mc = mcnemar_exact(g.a, g.b)
            bs = paired_bootstrap_difference(g.a, g.b)
            out.append({
                "intervention_a": a_name,
                "intervention_b": b_name,
                "model_id": model,
                "benchmark": bench,
                "topology": topology,
                "probe_kind": kind,
                **mc,
                **{f"bootstrap_{k}": v for k, v in bs.items()},
            })
    return pd.DataFrame(out)


def fit_glm(probes: pd.DataFrame, out_path: Path) -> None:
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception as e:
        out_path.write_text(f"statsmodels unavailable: {e}\n", encoding="utf-8")
        return
    d = probes[probes.probe_kind == "indirect"].copy()
    if d.empty:
        out_path.write_text("No indirect probes available.\n", encoding="utf-8")
        return
    formula = "target_positive ~ C(topology) * C(intervention) + C(model_id) + C(benchmark)"
    try:
        model = smf.glm(formula=formula, data=d, family=sm.families.Binomial())
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": d["unit_id"]})
        out_path.write_text(fit.summary().as_text(), encoding="utf-8")
    except Exception as e:
        out_path.write_text(f"GLM failed: {type(e).__name__}: {e}\n", encoding="utf-8")


def fit_mixed(probes: pd.DataFrame, out_path: Path) -> None:
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as e:
        out_path.write_text(f"mixed-effects model unavailable: {e}\n", encoding="utf-8")
        return
    d = probes[probes.probe_kind == "indirect"].copy()
    if d.empty:
        out_path.write_text("No indirect probes available.\n", encoding="utf-8")
        return
    formula = "target_positive ~ C(topology) * C(intervention) + C(model_id) + C(benchmark)"
    vc = {"unit": "0 + C(unit_id)", "probe": "0 + C(probe_id)"}
    try:
        model = BinomialBayesMixedGLM.from_formula(formula, vc, d)
        fit = model.fit_vb()
        out_path.write_text(str(fit.summary()), encoding="utf-8")
    except Exception as e:
        out_path.write_text(f"mixed-effects fit failed: {type(e).__name__}: {e}\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    ap.add_argument("--mixed-effects", action="store_true")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    mdir = cfg.paths.resolve(cfg.paths.outputs_dir) / "metrics"
    adir = cfg.paths.resolve(cfg.paths.outputs_dir) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)

    probes = pd.read_csv(mdir / "probe_scores.csv")
    run = pd.read_csv(mdir / "run_metrics.csv")
    run2 = add_cd_and_ramp(run)
    run2.to_csv(adir / "run_metrics_with_cd_ramp.csv", index=False)
    aggregate_probe_rates(probes).to_csv(adir / "aggregate_rates_with_ci.csv", index=False)
    paired_comparisons(probes).to_csv(adir / "paired_tests.csv", index=False)
    fit_glm(probes, adir / "logistic_glm.txt")
    if args.mixed_effects:
        fit_mixed(probes, adir / "mixed_effects_logistic.txt")


if __name__ == "__main__":
    main()
