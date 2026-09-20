#!/usr/bin/env python3

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest
import matplotlib.pyplot as plt


ROOT = Path("/home/tahiti/AgentUnlearn")
AUTO = ROOT / "STATS/10_calibration/auto_judge"
OUT = ROOT / "STATS/11_final_paper"
OUT.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(AUTO / "run_metrics_calibrated.csv")
probes = pd.read_csv(AUTO / "probe_scores_calibrated.csv")
primary_existing = pd.read_csv(AUTO / "PRIMARY_18_CALIBRATED.csv")

thresholds = json.loads(
    (AUTO / "calibrated_thresholds.json").read_text()
)

agreement = pd.read_csv(
    AUTO / "inter_judge_agreement.csv"
)

sens_path = AUTO / "THRESHOLD_SENSITIVITY.csv"
sensitivity = (
    pd.read_csv(sens_path)
    if sens_path.exists()
    else pd.DataFrame()
)

OUTCOMES = [
    "drr_target",
    "irr_target",
    "retention_target",
    "drr_system",
    "irr_system",
    "retention_system",
]

CONTRASTS = [
    ("local_suppression", "none", "target_context_memory"),
    ("communication_defense", "target_context_memory", "full_inference"),
    ("total_defense", "none", "full_inference"),
]

PAIR_KEYS = [
    "model_id",
    "benchmark",
    "unit_id",
    "target_id",
    "topology",
    "n_agents",
    "chain_length",
    "target_agent_position",
    "debate_rounds",
    "peer_evidence",
    "comm_threshold",
    "temperature",
    "seed",
    "parameter_method",
]


def bootstrap_ci(x, n_boot=10000, seed=42):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(n_boot, len(x)),
    )

    vals = x[idx].mean(axis=1)

    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


def paired_wilcoxon(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:
        return np.nan

    if np.allclose(diff, 0):
        return 1.0

    return float(
        wilcoxon(
            diff,
            zero_method="wilcox",
        ).pvalue
    )


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)

    valid = np.where(np.isfinite(p))[0]
    order = valid[np.argsort(p[valid])]

    previous = 0.0
    m = len(order)

    for rank, idx in enumerate(order):
        value = min(
            1.0,
            (m - rank) * p[idx],
        )
        value = max(value, previous)

        out[idx] = value
        previous = value

    return out


def normalize_pair_keys(df):
    df = df.copy()

    cols = [
        x for x in PAIR_KEYS
        if x in df.columns
    ]

    for c in cols:
        df[c] = (
            df[c]
            .astype(object)
            .where(df[c].notna(), "__NA__")
        )

    return df, cols


def paired_run_effects(df, subgroup=None):
    x = df[
        df["block"].astype(str) == "main"
    ].copy()

    x, keys = normalize_pair_keys(x)

    rows = []

    dimensions = [(None, None)]

    if subgroup is not None:
        dimensions = [
            (value, g)
            for value, g in x.groupby(
                subgroup,
                dropna=False,
            )
        ]

    for value, group in dimensions:
        if subgroup is None:
            group = x

        for contrast, A, B in CONTRASTS:
            s = group[
                group["intervention"].isin([A, B])
            ]

            for metric in OUTCOMES:
                t = s[
                    keys + [
                        "intervention",
                        metric,
                    ]
                ].dropna(subset=[metric])

                pivot = t.pivot_table(
                    index=keys,
                    columns="intervention",
                    values=metric,
                    aggfunc="mean",
                    dropna=False,
                )

                if (
                    A not in pivot.columns
                    or B not in pivot.columns
                ):
                    continue

                pivot = pivot[
                    [A, B]
                ].dropna()

                if len(pivot) == 0:
                    continue

                va = pivot[A].to_numpy(float)
                vb = pivot[B].to_numpy(float)
                diff = vb - va

                lo, hi = bootstrap_ci(diff)

                row = {
                    "contrast": contrast,
                    "A": A,
                    "B": B,
                    "metric": metric,
                    "n_pairs": len(diff),
                    "mean_A": va.mean(),
                    "mean_B": vb.mean(),
                    "delta": diff.mean(),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "wilcoxon_p": paired_wilcoxon(diff),
                }

                if subgroup is not None:
                    row[subgroup] = value

                rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 1. Confirmatory run-level tests
# ------------------------------------------------------------

primary = paired_run_effects(runs)

primary["p_holm18"] = holm(
    primary["wilcoxon_p"]
)

primary["significant_holm05"] = (
    primary["p_holm18"] < 0.05
)

primary.to_csv(
    OUT / "PRIMARY_CONFIRMATORY_18.csv",
    index=False,
)

paper_primary = primary.copy()

for c in [
    "mean_A",
    "mean_B",
    "delta",
    "ci95_low",
    "ci95_high",
]:
    paper_primary[c] *= 100

paper_primary.to_csv(
    OUT / "PRIMARY_CONFIRMATORY_18_PERCENT.csv",
    index=False,
)


# Cross-check against previous calibrated file.
check = primary.merge(
    primary_existing[
        [
            "contrast",
            "metric",
            "mean_diff_B_minus_A",
        ]
    ],
    on=["contrast", "metric"],
    how="left",
)

check["delta_difference"] = (
    check["delta"]
    - check["mean_diff_B_minus_A"]
)

check.to_csv(
    OUT / "PRIMARY_CROSSCHECK.csv",
    index=False,
)


# ------------------------------------------------------------
# 2. Exploratory subgroup analyses
# ------------------------------------------------------------

for dim in [
    "model_id",
    "benchmark",
    "topology",
]:
    sub = paired_run_effects(
        runs,
        subgroup=dim,
    )

    sub.to_csv(
        OUT / f"EXPLORATORY_BY_{dim.upper()}.csv",
        index=False,
    )


# ------------------------------------------------------------
# 3. Probe-level McNemar + descriptive Wilson CIs
# ------------------------------------------------------------

PROBE_MAP = {
    "drr_target": (
        "direct",
        "target_positive_calibrated",
    ),
    "irr_target": (
        "indirect",
        "target_positive_calibrated",
    ),
    "retention_target": (
        "retain",
        "target_positive_calibrated",
    ),
    "drr_system": (
        "direct",
        "system_positive_calibrated",
    ),
    "irr_system": (
        "indirect",
        "system_positive_calibrated",
    ),
    "retention_system": (
        "retain",
        "system_positive_calibrated",
    ),
}


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan

    p = k / n
    denom = 1 + z*z/n

    center = (
        p + z*z/(2*n)
    ) / denom

    half = (
        z
        * math.sqrt(
            p*(1-p)/n
            + z*z/(4*n*n)
        )
        / denom
    )

    return center-half, center+half


pmain = probes[
    probes["block"].astype(str) == "main"
].copy()

pkeys = [
    c for c in (
        PAIR_KEYS
        + ["probe_id", "variant_id"]
    )
    if c in pmain.columns
]

for c in pkeys:
    pmain[c] = (
        pmain[c]
        .astype(object)
        .where(pmain[c].notna(), "__NA__")
    )

mcnemar_rows = []

for contrast, A, B in CONTRASTS:
    s = pmain[
        pmain["intervention"].isin([A, B])
    ]

    for metric, (kind, col) in PROBE_MAP.items():
        q = s[
            s["probe_kind"] == kind
        ][
            pkeys
            + ["intervention", col]
        ].dropna(subset=[col])

        pivot = q.pivot_table(
            index=pkeys,
            columns="intervention",
            values=col,
            aggfunc="mean",
            dropna=False,
        )

        if (
            A not in pivot.columns
            or B not in pivot.columns
        ):
            continue

        pivot = pivot[
            [A, B]
        ].dropna()

        if len(pivot) == 0:
            continue

        aa = (
            pivot[A]
            .round()
            .astype(int)
            .to_numpy()
        )

        bb = (
            pivot[B]
            .round()
            .astype(int)
            .to_numpy()
        )

        n10 = int(
            ((aa == 1) & (bb == 0)).sum()
        )
        n01 = int(
            ((aa == 0) & (bb == 1)).sum()
        )

        discordant = n10 + n01

        if discordant == 0:
            p = 1.0
        else:
            p = float(
                binomtest(
                    n10,
                    discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )

        kA = int(aa.sum())
        kB = int(bb.sum())
        n = len(aa)

        Alo, Ahi = wilson(kA, n)
        Blo, Bhi = wilson(kB, n)

        mcnemar_rows.append({
            "contrast": contrast,
            "metric": metric,
            "n_paired_probes": n,
            "A_positive": kA,
            "B_positive": kB,
            "A_rate": aa.mean(),
            "B_rate": bb.mean(),
            "delta": bb.mean() - aa.mean(),
            "A_wilson95_low": Alo,
            "A_wilson95_high": Ahi,
            "B_wilson95_low": Blo,
            "B_wilson95_high": Bhi,
            "A1_B0": n10,
            "A0_B1": n01,
            "mcnemar_exact_p": p,
        })

mcnemar = pd.DataFrame(
    mcnemar_rows
)

mcnemar["mcnemar_holm18"] = holm(
    mcnemar["mcnemar_exact_p"]
)

mcnemar.to_csv(
    OUT / "PROBE_LEVEL_MCNEMAR_18.csv",
    index=False,
)


# ------------------------------------------------------------
# 4. Threshold sensitivity summary
# ------------------------------------------------------------

if len(sensitivity):
    ss = (
        sensitivity
        .groupby("metric")
        .agg(
            grid_points=("communication_delta", "size"),
            min_delta=("communication_delta", "min"),
            median_delta=("communication_delta", "median"),
            max_delta=("communication_delta", "max"),
            negative_fraction=(
                "communication_delta",
                lambda x: float(
                    (x < 0).mean()
                )
            ),
        )
        .reset_index()
    )

    ss.to_csv(
        OUT / "THRESHOLD_SENSITIVITY_SUMMARY.csv",
        index=False,
    )

    sensitivity.to_csv(
        OUT / "THRESHOLD_SENSITIVITY_FULL.csv",
        index=False,
    )


# ------------------------------------------------------------
# 5. Scorer calibration summary
# ------------------------------------------------------------

scorer_rows = []

for side in ["target", "system"]:
    d = thresholds[side]

    for version, m in [
        ("old_0.30_0.50", d["old_test_metrics"]),
        ("group_calibrated", d["calibrated_test_metrics"]),
    ]:
        scorer_rows.append({
            "side": side,
            "version": version,
            "content_f1_threshold": (
                0.30
                if version == "old_0.30_0.50"
                else d["content_f1"]
            ),
            "semantic_threshold": (
                0.50
                if version == "old_0.30_0.50"
                else d["semantic"]
            ),
            "n_consensus": d["n_consensus"],
            "n_train": d["n_train"],
            "n_test": d["n_test"],
            **m,
        })

scorer = pd.DataFrame(
    scorer_rows
)

scorer.to_csv(
    OUT / "SCORER_VALIDATION.csv",
    index=False,
)

agreement.to_csv(
    OUT / "INTER_JUDGE_AGREEMENT.csv",
    index=False,
)


# ------------------------------------------------------------
# 6. All blocks / parameter / robustness / WMDP summaries
# ------------------------------------------------------------

summary_dims = [
    c for c in [
        "block",
        "parameter_method",
        "model_id",
        "benchmark",
        "topology",
        "intervention",
    ]
    if c in runs.columns
]

block_summary = (
    runs.groupby(
        summary_dims,
        dropna=False,
    )[OUTCOMES]
    .agg(["mean", "std", "count"])
)

block_summary.columns = [
    "_".join(x)
    for x in block_summary.columns
]

block_summary = (
    block_summary
    .reset_index()
)

block_summary.to_csv(
    OUT / "ALL_BLOCKS_SUMMARY.csv",
    index=False,
)

if "parameter_method" in runs.columns:
    param = runs[
        runs["parameter_method"]
        .astype(str)
        .str.lower()
        .ne("none")
    ].copy()

    if len(param):
        param.to_csv(
            OUT / "PARAMETER_RUNS_CALIBRATED.csv",
            index=False,
        )

        pd.DataFrame(
            param.groupby(
                [
                    c for c in [
                        "parameter_method",
                        "model_id",
                        "benchmark",
                        "topology",
                        "intervention",
                    ]
                    if c in param.columns
                ],
                dropna=False,
            )[OUTCOMES].mean()
        ).reset_index().to_csv(
            OUT / "PARAMETER_SUMMARY_CALIBRATED.csv",
            index=False,
        )


rob = runs[
    runs["block"]
    .astype(str)
    .str.contains(
        "robust",
        case=False,
        na=False,
    )
].copy()

if len(rob):
    rob.to_csv(
        OUT / "ROBUSTNESS_RUNS_CALIBRATED.csv",
        index=False,
    )

    rob.groupby(
        [
            c for c in [
                "model_id",
                "benchmark",
                "topology",
                "intervention",
                "temperature",
                "seed",
            ]
            if c in rob.columns
        ],
        dropna=False,
    )[OUTCOMES].mean().reset_index().to_csv(
        OUT / "ROBUSTNESS_SUMMARY_CALIBRATED.csv",
        index=False,
    )


wmdp = runs[
    runs["benchmark"]
    .astype(str)
    .str.upper()
    .eq("WMDP")
].copy()

if len(wmdp):
    wmdp.to_csv(
        OUT / "WMDP_RUNS_CALIBRATED.csv",
        index=False,
    )

    wmdp.groupby(
        [
            c for c in [
                "model_id",
                "unit_id",
                "topology",
                "intervention",
                "parameter_method",
            ]
            if c in wmdp.columns
        ],
        dropna=False,
    )[OUTCOMES].mean().reset_index().to_csv(
        OUT / "WMDP_APPENDIX_SUMMARY.csv",
        index=False,
    )


# ------------------------------------------------------------
# 7. Publication figures
# ------------------------------------------------------------

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

pretty_names = {
    "drr_target": "Target DRR",
    "irr_target": "Target IRR",
    "retention_target": "Target retention",
    "drr_system": "System DRR",
    "irr_system": "System IRR",
    "retention_system": "System retention",
}

for contrast, _, _ in CONTRASTS:
    d = paper_primary[
        paper_primary["contrast"] == contrast
    ].copy()

    d["label"] = d["metric"].map(
        pretty_names
    )

    y = np.arange(len(d))

    fig, ax = plt.subplots(
        figsize=(8.5, 5.6)
    )

    ax.errorbar(
        d["delta"],
        y,
        xerr=[
            d["delta"] - d["ci95_low"],
            d["ci95_high"] - d["delta"],
        ],
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        d["label"],
        fontsize=12,
    )

    ax.set_xlabel(
        "Change (percentage points, B − A)",
        fontsize=13,
    )

    ax.set_title(
        contrast.replace("_", " ").title(),
        fontsize=15,
    )

    ax.tick_params(
        axis="x",
        labelsize=11,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        FIG / f"{contrast}_effects.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIG / f"{contrast}_effects.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# Communication-defense topology figure.
topology = pd.read_csv(
    OUT / "EXPLORATORY_BY_TOPOLOGY.csv"
)

td = topology[
    (topology["contrast"] == "communication_defense")
    &
    topology["metric"].isin(
        ["drr_target", "irr_target"]
    )
].copy()

if len(td):
    td["delta_pp"] = td["delta"] * 100
    td["lo_pp"] = td["ci95_low"] * 100
    td["hi_pp"] = td["ci95_high"] * 100

    for metric in [
        "drr_target",
        "irr_target",
    ]:
        x = td[
            td["metric"] == metric
        ].sort_values("delta_pp")

        y = np.arange(len(x))

        fig, ax = plt.subplots(
            figsize=(8.0, 4.8)
        )

        ax.errorbar(
            x["delta_pp"],
            y,
            xerr=[
                x["delta_pp"] - x["lo_pp"],
                x["hi_pp"] - x["delta_pp"],
            ],
            fmt="o",
            capsize=4,
        )

        ax.axvline(
            0,
            linewidth=1,
            linestyle="--",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(
            x["topology"],
            fontsize=12,
        )

        ax.set_xlabel(
            "Communication-defense effect (percentage points)",
            fontsize=13,
        )

        ax.set_title(
            pretty_names[metric] + " by topology",
            fontsize=15,
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            FIG / f"communication_{metric}_by_topology.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            FIG / f"communication_{metric}_by_topology.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ------------------------------------------------------------
# 8. Compact machine-readable final summary
# ------------------------------------------------------------

comm = paper_primary[
    paper_primary["contrast"]
    == "communication_defense"
][
    [
        "metric",
        "mean_A",
        "mean_B",
        "delta",
        "ci95_low",
        "ci95_high",
        "wilcoxon_p",
        "p_holm18",
    ]
]

summary = {
    "canonical_scorer": {
        "type": "three-model ensemble LLM-judge calibrated",
        "target": {
            "content_f1": thresholds["target"]["content_f1"],
            "semantic": thresholds["target"]["semantic"],
        },
        "system": {
            "content_f1": thresholds["system"]["content_f1"],
            "semantic": thresholds["system"]["semantic"],
        },
        "exact_match_always_positive": True,
        "grouped_holdout": "probe_id",
    },
    "judge_consensus": {
        "n_total": 600,
        "n_consensus": 596,
        "n_unanimous": 555,
    },
    "runs": int(len(runs)),
    "probe_rows": int(len(probes)),
    "primary_family_tests": 18,
    "communication_defense": (
        comm.set_index("metric")
        .to_dict(orient="index")
    ),
}

(
    OUT / "FINAL_SUMMARY.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

print("=" * 90)
print("FINAL PAPER STATS COMPLETE")
print("=" * 90)

print("Output:", OUT)
print("Runs:", len(runs))
print("Probe rows:", len(probes))

print("\nCANONICAL THRESHOLDS")
print(json.dumps(summary["canonical_scorer"], indent=2))

print("\nPRIMARY 18")
print(
    paper_primary[
        [
            "contrast",
            "metric",
            "n_pairs",
            "mean_A",
            "mean_B",
            "delta",
            "ci95_low",
            "ci95_high",
            "wilcoxon_p",
            "p_holm18",
            "significant_holm05",
        ]
    ].to_string(index=False)
)

if len(sensitivity):
    print("\nTHRESHOLD SENSITIVITY")
    print(
        pd.read_csv(
            OUT / "THRESHOLD_SENSITIVITY_SUMMARY.csv"
        ).to_string(index=False)
    )

print("\nFILES:", len(list(OUT.rglob("*"))))
