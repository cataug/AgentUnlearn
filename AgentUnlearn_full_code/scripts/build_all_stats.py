#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


ROOT = Path("/home/tahiti/AgentUnlearn")
CODE = ROOT / "AgentUnlearn_full_code"
FULL = ROOT / "outputs/full_plan"
METRICS = FULL / "metrics"
CHECKPOINTS = FULL / "checkpoints"
RAW = FULL / "raw"
STATS = ROOT / "STATS"

STATS.mkdir(parents=True, exist_ok=True)

for d in [
    "00_snapshot",
    "01_inventory",
    "02_run_level",
    "03_probe_level",
    "04_intervention_tests",
    "05_parameter_unlearning",
    "06_training",
    "07_robustness",
    "08_tables",
]:
    (STATS / d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def copy_tree(src: Path, dst: Path):
    if src.exists():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
        )


def flatten_dict(d, prefix=""):
    out = {}

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)

        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif not isinstance(v, (list, tuple)):
            out[key] = v

    return out


CONFIG_NUMERIC = {
    "seed",
    "temperature",
    "comm_threshold",
    "communication_threshold",
    "n_agents",
    "num_agents",
    "agent_count",
    "target_position",
    "rounds",
    "debate_rounds",
    "max_rounds",
    "max_new_tokens",
    "prompt_variants",
    "base_direct_probes",
    "base_indirect_probes",
    "base_retain_probes",
    "max_target_evidence",
    "max_retain_evidence",
}


def metric_columns(df: pd.DataFrame):
    result = []

    for c in df.columns:
        if c in CONFIG_NUMERIC:
            continue

        if c.lower().endswith("_id"):
            continue

        if c.lower() in {
            "id",
            "run_id",
            "training_id",
            "variant_id",
            "probe_id",
        }:
            continue

        if pd.api.types.is_bool_dtype(df[c]):
            result.append(c)

        elif pd.api.types.is_numeric_dtype(df[c]):
            result.append(c)

    return result


def aggregate_long(
    df: pd.DataFrame,
    groups: list[str],
    metrics: list[str],
):
    rows = []

    if not metrics:
        return pd.DataFrame()

    if groups:
        iterator = df.groupby(
            groups,
            dropna=False,
            sort=True,
        )
    else:
        iterator = [((), df)]

    for key, g in iterator:

        if not isinstance(key, tuple):
            key = (key,)

        base = {
            col: val
            for col, val in zip(groups, key)
        }

        for metric in metrics:

            x = (
                pd.to_numeric(
                    g[metric],
                    errors="coerce",
                )
                .dropna()
                .astype(float)
            )

            if len(x) == 0:
                continue

            n = len(x)
            mean = float(x.mean())
            std = float(x.std(ddof=1)) if n > 1 else np.nan
            sem = (
                std / math.sqrt(n)
                if n > 1 and np.isfinite(std)
                else np.nan
            )

            row = dict(base)

            row.update({
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "median": float(x.median()),
                "q25": float(x.quantile(0.25)),
                "q75": float(x.quantile(0.75)),
                "min": float(x.min()),
                "max": float(x.max()),
                "ci95_low": (
                    mean - 1.96 * sem
                    if np.isfinite(sem)
                    else np.nan
                ),
                "ci95_high": (
                    mean + 1.96 * sem
                    if np.isfinite(sem)
                    else np.nan
                ),
            })

            rows.append(row)

    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan

    p = k / n
    den = 1 + z * z / n
    center = (
        p + z * z / (2 * n)
    ) / den

    half = (
        z
        * math.sqrt(
            p * (1 - p) / n
            + z * z / (4 * n * n)
        )
        / den
    )

    return center - half, center + half


def bootstrap_mean_ci(x, n_boot=2000, seed=42):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    vals = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        vals[i] = rng.choice(
            x,
            size=len(x),
            replace=True,
        ).mean()

    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)

    valid = np.where(np.isfinite(p))[0]

    if len(valid) == 0:
        return out

    order = valid[
        np.argsort(p[valid])
    ]

    m = len(order)
    previous = 0.0

    for rank, idx in enumerate(order):
        adjusted = min(
            1.0,
            (m - rank) * p[idx],
        )

        adjusted = max(
            previous,
            adjusted,
        )

        out[idx] = adjusted
        previous = adjusted

    return out


def join_plan(df: pd.DataFrame, plan: pd.DataFrame):
    if "run_id" not in df.columns:
        return df.copy()

    result = df.merge(
        plan,
        on="run_id",
        how="left",
        suffixes=("", "__plan"),
    )

    for c in list(result.columns):
        if not c.endswith("__plan"):
            continue

        base = c[:-6]

        if base not in result.columns:
            result = result.rename(
                columns={c: base}
            )
        else:
            result = result.drop(
                columns=[c]
            )

    return result


# ---------------------------------------------------------------------
# 00. Exact snapshots
# ---------------------------------------------------------------------

copy_tree(
    METRICS,
    STATS / "00_snapshot" / "metrics",
)

copy_tree(
    FULL / "plan_snapshot",
    STATS / "00_snapshot" / "plan",
)

for p in [
    CODE / "config/production.json",
    FULL / "orchestrator_state.json",
    ROOT / "manifests/runs_all.csv",
    ROOT / "manifests/training_parameter_unlearning.csv",
    ROOT / "manifests/benchmark_units_manifest.csv",
]:
    if p.exists():
        shutil.copy2(
            p,
            STATS / "00_snapshot" / p.name,
        )


# ---------------------------------------------------------------------
# 01. Inventory
# ---------------------------------------------------------------------

inventory = []

for p in FULL.rglob("*"):
    if not p.is_file():
        continue

    inventory.append({
        "relative_path": str(
            p.relative_to(FULL)
        ),
        "size_bytes": p.stat().st_size,
        "size_mib": (
            p.stat().st_size
            / 1024**2
        ),
    })

pd.DataFrame(inventory).to_csv(
    STATS / "01_inventory/full_plan_files.csv",
    index=False,
)

raw_files = sorted(
    RAW.glob("*.json")
)

pd.DataFrame([
    {
        "run_id": p.stem,
        "path": str(p),
        "size_bytes": p.stat().st_size,
    }
    for p in raw_files
]).to_csv(
    STATS / "01_inventory/raw_runs.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Resolve fixed experiment plan
# ---------------------------------------------------------------------

plan_candidates = [
    FULL / "plan_snapshot/runs_all.csv",
    ROOT / "manifests/runs_all.csv",
]

plan_path = next(
    p
    for p in plan_candidates
    if p.exists()
)

plan = pd.read_csv(plan_path)

plan["raw_complete"] = plan["run_id"].astype(str).map(
    lambda x: (
        RAW / f"{x}.json"
    ).exists()
)

plan.to_csv(
    STATS / "01_inventory/experiment_plan_with_status.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Discover ALL scoring outputs
# ---------------------------------------------------------------------

metric_csvs = sorted(
    METRICS.rglob("*.csv")
)

metric_index = []

for p in metric_csvs:
    try:
        df = pd.read_csv(p)

        metric_index.append({
            "file": str(p.relative_to(METRICS)),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": " | ".join(df.columns),
        })

    except Exception as e:
        metric_index.append({
            "file": str(p.relative_to(METRICS)),
            "rows": np.nan,
            "columns": np.nan,
            "column_names": f"ERROR: {e}",
        })

pd.DataFrame(metric_index).to_csv(
    STATS / "01_inventory/metric_files.csv",
    index=False,
)

run_tables = [
    p for p in metric_csvs
    if "run_metrics" in p.name.lower()
]

probe_tables = [
    p for p in metric_csvs
    if "probe_scores" in p.name.lower()
]

trace_tables = [
    p for p in metric_csvs
    if "trace" in p.name.lower()
]


# ---------------------------------------------------------------------
# 02. Run-level statistics for EVERY run_metrics file
# ---------------------------------------------------------------------

run_joined_tables = {}

group_specs = {
    "overall": [],
    "by_block": ["block"],
    "by_model": ["model_id"],
    "by_benchmark": ["benchmark"],
    "by_unit": ["unit_id"],
    "by_topology": ["topology"],
    "by_intervention": ["intervention"],
    "by_parameter_method": ["parameter_method"],

    "model_benchmark":
        ["model_id", "benchmark"],

    "benchmark_topology":
        ["benchmark", "topology"],

    "benchmark_intervention":
        ["benchmark", "intervention"],

    "topology_intervention":
        ["topology", "intervention"],

    "model_topology_intervention":
        ["model_id", "topology", "intervention"],

    "benchmark_topology_intervention":
        ["benchmark", "topology", "intervention"],

    "model_benchmark_topology_intervention":
        [
            "model_id",
            "benchmark",
            "topology",
            "intervention",
        ],
}


for p in run_tables:

    df = pd.read_csv(p)
    joined = join_plan(df, plan)

    tag = p.stem

    run_joined_tables[tag] = joined

    outdir = (
        STATS
        / "02_run_level"
        / tag
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joined.to_csv(
        outdir / "joined_with_plan.csv",
        index=False,
    )

    metrics = metric_columns(df)

    (outdir / "metric_columns.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    for name, groups in group_specs.items():

        groups = [
            x
            for x in groups
            if x in joined.columns
        ]

        if name != "overall" and not groups:
            continue

        agg = aggregate_long(
            joined,
            groups,
            metrics,
        )

        agg.to_csv(
            outdir / f"{name}.csv",
            index=False,
        )


# Preferred run table = content-aware one if available.
preferred_run_tag = None

if run_tables:
    preferred_path = sorted(
        run_tables,
        key=lambda p: (
            "content" in p.name.lower(),
            p.stat().st_mtime,
        ),
        reverse=True,
    )[0]

    preferred_run_tag = preferred_path.stem
    RUN = run_joined_tables[
        preferred_run_tag
    ]

    shutil.copy2(
        preferred_path,
        STATS
        / "02_run_level"
        / "PREFERRED_RUN_METRICS.csv",
    )
else:
    RUN = None


# ---------------------------------------------------------------------
# 03. Probe-level statistics for EVERY probe_scores file
# ---------------------------------------------------------------------

for p in probe_tables:

    df = pd.read_csv(p)
    joined = join_plan(df, plan)

    tag = p.stem

    outdir = (
        STATS
        / "03_probe_level"
        / tag
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joined.to_csv(
        outdir / "joined_with_plan.csv",
        index=False,
    )

    metrics = metric_columns(df)

    probe_groupings = {
        "overall": [],
        "by_probe_type":
            ["probe_type"],

        "benchmark_probe":
            ["benchmark", "probe_type"],

        "intervention_probe":
            ["intervention", "probe_type"],

        "topology_intervention_probe":
            [
                "topology",
                "intervention",
                "probe_type",
            ],

        "benchmark_intervention_probe":
            [
                "benchmark",
                "intervention",
                "probe_type",
            ],

        "model_benchmark_intervention_probe":
            [
                "model_id",
                "benchmark",
                "intervention",
                "probe_type",
            ],
    }

    for name, groups in probe_groupings.items():

        groups = [
            x for x in groups
            if x in joined.columns
        ]

        if name != "overall" and not groups:
            continue

        aggregate_long(
            joined,
            groups,
            metrics,
        ).to_csv(
            outdir / f"{name}.csv",
            index=False,
        )

    # Wilson CIs for every actual binary outcome column.
    binary_cols = []

    for c in metrics:
        vals = pd.to_numeric(
            joined[c],
            errors="coerce",
        ).dropna()

        uniq = set(
            vals.unique().tolist()
        )

        if (
            len(uniq) > 0
            and uniq.issubset({0, 1})
        ):
            binary_cols.append(c)

    rows = []

    groups = [
        x
        for x in [
            "benchmark",
            "topology",
            "intervention",
            "probe_type",
        ]
        if x in joined.columns
    ]

    iterator = (
        joined.groupby(
            groups,
            dropna=False,
        )
        if groups
        else [((), joined)]
    )

    for key, g in iterator:

        if not isinstance(key, tuple):
            key = (key,)

        base = {
            c: v
            for c, v in zip(groups, key)
        }

        for c in binary_cols:

            x = pd.to_numeric(
                g[c],
                errors="coerce",
            ).dropna()

            n = len(x)

            if n == 0:
                continue

            k = int(x.sum())
            lo, hi = wilson(k, n)

            rows.append({
                **base,
                "metric": c,
                "n": n,
                "positive": k,
                "rate": k / n,
                "wilson95_low": lo,
                "wilson95_high": hi,
            })

    pd.DataFrame(rows).to_csv(
        outdir / "binary_wilson95.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# 04. Paired intervention tests
# ---------------------------------------------------------------------

def paired_comparison(
    frame,
    a,
    b,
    metrics,
    label,
    stratify=None,
):
    if "intervention" not in frame.columns:
        return []

    x = frame[
        frame["intervention"].isin([a, b])
    ].copy()

    if len(x) == 0:
        return []

    candidates = [
        "block",
        "model_id",
        "benchmark",
        "unit_id",
        "target_id",
        "topology",
        "parameter_method",
        "seed",
        "temperature",
        "n_agents",
        "num_agents",
        "agent_count",
        "target_position",
        "comm_threshold",
        "debate_rounds",
        "rounds",
    ]

    keys = []

    for c in candidates:
        if (
            c in x.columns
            and not x[c].isna().all()
            and c != stratify
        ):
            keys.append(c)

    if not keys:
        return []

    for c in keys:
        x[c] = (
            x[c]
            .astype(object)
            .where(
                x[c].notna(),
                "__NA__",
            )
        )

    strata = (
        [(None, x)]
        if stratify is None
        else list(
            x.groupby(
                stratify,
                dropna=False,
            )
        )
    )

    rows = []

    for stratum, g in strata:

        for metric in metrics:

            if metric not in g.columns:
                continue

            t = (
                g[
                    keys
                    + [
                        "intervention",
                        metric,
                    ]
                ]
                .copy()
            )

            t[metric] = pd.to_numeric(
                t[metric],
                errors="coerce",
            )

            t = t.dropna(
                subset=[metric]
            )

            if len(t) == 0:
                continue

            # Average duplicates within an exact experimental key.
            t = (
                t.groupby(
                    keys + ["intervention"],
                    dropna=False,
                )[metric]
                .mean()
                .reset_index()
            )

            w = t.pivot_table(
                index=keys,
                columns="intervention",
                values=metric,
                aggfunc="mean",
            )

            if (
                a not in w.columns
                or b not in w.columns
            ):
                continue

            w = w[
                [a, b]
            ].dropna()

            if len(w) == 0:
                continue

            va = w[a].to_numpy(
                dtype=float
            )

            vb = w[b].to_numpy(
                dtype=float
            )

            diff = vb - va

            lo, hi = bootstrap_mean_ci(
                diff
            )

            p = np.nan

            if HAVE_SCIPY and len(diff) > 0:
                try:
                    if np.allclose(
                        diff,
                        0,
                    ):
                        p = 1.0
                    else:
                        p = float(
                            wilcoxon(
                                diff,
                                zero_method="wilcox",
                                alternative="two-sided",
                            ).pvalue
                        )
                except Exception:
                    p = np.nan

            row = {
                "contrast": label,
                "A": a,
                "B": b,
                "metric": metric,
                "n_pairs": len(diff),
                "mean_A": float(
                    np.mean(va)
                ),
                "mean_B": float(
                    np.mean(vb)
                ),
                "mean_diff_B_minus_A": float(
                    np.mean(diff)
                ),
                "median_diff": float(
                    np.median(diff)
                ),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
                "wilcoxon_p": p,
            }

            if stratify is not None:
                row[stratify] = stratum

            rows.append(row)

    return rows


if RUN is not None:

    metrics = metric_columns(
        RUN
    )

    main = RUN.copy()

    if "block" in main.columns:
        mask = (
            main["block"]
            .astype(str)
            .str.lower()
            .str.contains("main")
        )

        if mask.any():
            main = main[mask].copy()

    key_contrasts = [
        (
            "none",
            "target_context_memory",
            "local_suppression",
        ),
        (
            "target_context_memory",
            "full_inference",
            "communication_defense",
        ),
        (
            "none",
            "full_inference",
            "total_defense",
        ),
    ]

    key_rows = []

    for a, b, label in key_contrasts:

        key_rows += paired_comparison(
            main,
            a,
            b,
            metrics,
            label,
            None,
        )

        for stratify in [
            "model_id",
            "benchmark",
            "topology",
        ]:
            if stratify in main.columns:
                key_rows += paired_comparison(
                    main,
                    a,
                    b,
                    metrics,
                    label,
                    stratify,
                )

    key_df = pd.DataFrame(
        key_rows
    )

    if len(key_df):
        key_df[
            "wilcoxon_p_holm"
        ] = holm_adjust(
            key_df[
                "wilcoxon_p"
            ].to_numpy()
        )

    key_df.to_csv(
        STATS
        / "04_intervention_tests"
        / "KEY_CAUSAL_COMPARISONS.csv",
        index=False,
    )

    # All pairwise main intervention comparisons.
    all_rows = []

    interventions = sorted(
        main[
            "intervention"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    for a, b in combinations(
        interventions,
        2,
    ):
        all_rows += paired_comparison(
            main,
            a,
            b,
            metrics,
            f"{a}__vs__{b}",
            None,
        )

    pair_df = pd.DataFrame(
        all_rows
    )

    if len(pair_df):
        pair_df[
            "wilcoxon_p_holm"
        ] = holm_adjust(
            pair_df[
                "wilcoxon_p"
            ].to_numpy()
        )

    pair_df.to_csv(
        STATS
        / "04_intervention_tests"
        / "ALL_MAIN_INTERVENTION_PAIRS.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# 05. Parameter-unlearning statistics
# ---------------------------------------------------------------------

if RUN is not None:

    parameter = RUN.copy()

    if "parameter_method" in parameter.columns:
        parameter = parameter[
            parameter[
                "parameter_method"
            ]
            .fillna("none")
            .astype(str)
            != "none"
        ]

    parameter.to_csv(
        STATS
        / "05_parameter_unlearning"
        / "parameter_runs.csv",
        index=False,
    )

    param_groups = [
        "parameter_method",
        "model_id",
        "benchmark",
        "intervention",
        "topology",
    ]

    param_groups = [
        x
        for x in param_groups
        if x in parameter.columns
    ]

    aggregate_long(
        parameter,
        param_groups,
        metric_columns(parameter),
    ).to_csv(
        STATS
        / "05_parameter_unlearning"
        / "parameter_summary.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# 06. Training checkpoint statistics
# ---------------------------------------------------------------------

training_rows = []
loss_rows = []

if CHECKPOINTS.exists():

    for d in sorted(
        CHECKPOINTS.iterdir()
    ):
        if not d.is_dir():
            continue

        meta_path = (
            d / "metadata.json"
        )

        err_path = (
            d / "error.json"
        )

        row = {
            "training_id": d.name,
            "checkpoint_dir": str(d),
            "complete": meta_path.exists(),
            "error_present": err_path.exists(),
            "size_mib": sum(
                p.stat().st_size
                for p in d.rglob("*")
                if p.is_file()
            ) / 1024**2,
        }

        meta = safe_json(
            meta_path
        ) if meta_path.exists() else None

        if meta:
            hist = meta.get(
                "loss_history",
                []
            )

            clean = dict(meta)
            clean.pop(
                "loss_history",
                None,
            )

            row.update(
                flatten_dict(clean)
            )

            for h in hist:
                loss_rows.append({
                    "training_id": d.name,
                    **h,
                })

            if hist:
                first = hist[0]
                last = hist[-1]

                for k in [
                    "forget_loss",
                    "retain_loss",
                    "objective",
                ]:
                    if k in first:
                        row[
                            f"{k}_first"
                        ] = first.get(k)

                    if k in last:
                        row[
                            f"{k}_last"
                        ] = last.get(k)

        if err_path.exists():
            err = safe_json(
                err_path
            )

            if err:
                row[
                    "error_type"
                ] = err.get(
                    "error_type"
                )

                row[
                    "error"
                ] = err.get(
                    "error"
                )

        training_rows.append(
            row
        )


training_df = pd.DataFrame(
    training_rows
)

training_df.to_csv(
    STATS
    / "06_training"
    / "training_jobs.csv",
    index=False,
)

pd.DataFrame(
    loss_rows
).to_csv(
    STATS
    / "06_training"
    / "training_loss_history.csv",
    index=False,
)

if len(training_df):

    groups = [
        c
        for c in [
            "method",
            "model_id",
            "benchmark",
        ]
        if c in training_df.columns
    ]

    if groups:
        metrics = [
            c
            for c in [
                "forget_loss_first",
                "forget_loss_last",
                "retain_loss_first",
                "retain_loss_last",
                "objective_first",
                "objective_last",
                "size_mib",
            ]
            if c in training_df.columns
        ]

        aggregate_long(
            training_df,
            groups,
            metrics,
        ).to_csv(
            STATS
            / "06_training"
            / "training_summary.csv",
            index=False,
        )


# ---------------------------------------------------------------------
# 07. Robustness
# ---------------------------------------------------------------------

if RUN is not None:

    robustness = RUN.copy()

    if "block" in robustness.columns:

        mask = (
            robustness["block"]
            .astype(str)
            .str.lower()
            .str.contains("robust")
        )

        robustness = (
            robustness[mask]
            .copy()
        )

    robustness.to_csv(
        STATS
        / "07_robustness"
        / "robustness_runs.csv",
        index=False,
    )

    robust_groups = [
        c
        for c in [
            "model_id",
            "benchmark",
            "topology",
            "intervention",
            "temperature",
            "seed",
        ]
        if c in robustness.columns
    ]

    aggregate_long(
        robustness,
        robust_groups,
        metric_columns(robustness),
    ).to_csv(
        STATS
        / "07_robustness"
        / "robustness_summary.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# 08. Compact headline tables
# ---------------------------------------------------------------------

if RUN is not None:

    metrics = metric_columns(
        RUN
    )

    headline_groups = [
        c
        for c in [
            "model_id",
            "benchmark",
            "topology",
            "intervention",
        ]
        if c in RUN.columns
    ]

    aggregate_long(
        RUN,
        headline_groups,
        metrics,
    ).to_csv(
        STATS
        / "08_tables"
        / "FULL_FACTORIAL_LONG.csv",
        index=False,
    )

    if (
        "intervention" in RUN.columns
        and metrics
    ):
        table = (
            RUN.groupby(
                "intervention",
                dropna=False,
            )[metrics]
            .mean(
                numeric_only=True
            )
        )

        table.to_csv(
            STATS
            / "08_tables"
            / "INTERVENTION_MEANS_WIDE.csv"
        )

    if (
        "benchmark" in RUN.columns
        and "intervention" in RUN.columns
        and metrics
    ):
        table = (
            RUN.groupby(
                [
                    "benchmark",
                    "intervention",
                ],
                dropna=False,
            )[metrics]
            .mean(
                numeric_only=True
            )
        )

        table.to_csv(
            STATS
            / "08_tables"
            / "BENCHMARK_X_INTERVENTION.csv"
        )

    if (
        "topology" in RUN.columns
        and "intervention" in RUN.columns
        and metrics
    ):
        table = (
            RUN.groupby(
                [
                    "topology",
                    "intervention",
                ],
                dropna=False,
            )[metrics]
            .mean(
                numeric_only=True
            )
        )

        table.to_csv(
            STATS
            / "08_tables"
            / "TOPOLOGY_X_INTERVENTION.csv"
        )


# ---------------------------------------------------------------------
# Master summary
# ---------------------------------------------------------------------

summary = {
    "experiment": "AgentUnlearn",
    "runs_expected": int(
        len(plan)
    ),
    "runs_complete": int(
        plan["raw_complete"].sum()
    ),
    "training_jobs": int(
        len(training_df)
    ),
    "training_complete": int(
        training_df["complete"].sum()
    ) if len(training_df) else 0,
    "metric_csv_files": len(
        metric_csvs
    ),
    "run_metric_files": [
        p.name
        for p in run_tables
    ],
    "probe_score_files": [
        p.name
        for p in probe_tables
    ],
    "trace_files": [
        p.name
        for p in trace_tables
    ],
    "preferred_run_metrics":
        preferred_run_tag,
    "scipy_available":
        HAVE_SCIPY,
    "note": (
        "Content-aware thresholds 0.30/0.50 are "
        "the current automatic scoring thresholds; "
        "do not describe them as human-calibrated "
        "unless calibration has actually been completed."
    ),
}

(STATS / "SUMMARY.json").write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

readme = f"""# AgentUnlearn STATS

Complete statistical export of the finished AgentUnlearn experimental plan.

## Completion

- Inference runs: {summary['runs_complete']} / {summary['runs_expected']}
- Parameter-training jobs: {summary['training_complete']} / {summary['training_jobs']}
- Metric CSV files discovered: {summary['metric_csv_files']}
- Preferred run-level table: {summary['preferred_run_metrics']}

## Directory layout

- `00_snapshot/` — exact scoring/config/plan snapshot
- `01_inventory/` — experiment, raw-output and metric inventories
- `02_run_level/` — descriptive statistics for every run-level scoring table
- `03_probe_level/` — probe-level summaries and Wilson 95% intervals
- `04_intervention_tests/` — paired intervention contrasts, bootstrap CIs and Wilcoxon tests
- `05_parameter_unlearning/` — GA/LoRA evaluation statistics
- `06_training/` — checkpoint metadata and loss trajectories
- `07_robustness/` — stochastic robustness breakdowns
- `08_tables/` — compact factorial tables for paper analysis

## Main causal contrast

`target_context_memory` versus `full_inference` isolates the additional effect of
communication filtering after target-local context and memory are both removed.

See:

`04_intervention_tests/KEY_CAUSAL_COMPARISONS.csv`

## Scoring note

The current content-aware decision rule uses content-F1 >= 0.30 and semantic
similarity >= 0.50 (plus exact match where applicable). These thresholds should
be reported as automatic thresholds unless the human-validation calibration
pipeline is completed separately.
"""

(STATS / "README.md").write_text(
    readme
)

print()
print("=" * 100)
print("AGENTUNLEARN STATISTICS EXPORT COMPLETE")
print("=" * 100)
print("STATS:", STATS)
print()
print(json.dumps(summary, indent=2))
