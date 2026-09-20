#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/tahiti/AgentUnlearn")

AUTO = (
    ROOT
    / "STATS"
    / "10_calibration"
    / "auto_judge"
)

TECH = ROOT / "TECH_STATS"
FIG = ROOT / "FIGURES"

RAW = (
    ROOT
    / "outputs"
    / "full_plan"
    / "raw"
)

TECH.mkdir(
    parents=True,
    exist_ok=True,
)

FIG.mkdir(
    parents=True,
    exist_ok=True,
)


RUNS = (
    AUTO
    / "run_metrics_calibrated.csv"
)


runs = pd.read_csv(
    RUNS
)


# ============================================================
# Figure style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": 20,

    "axes.titlesize": 26,
    "axes.labelsize": 22,

    "xtick.labelsize": 17,
    "ytick.labelsize": 17,

    "legend.fontsize": 14,

    "axes.linewidth": 1.15,
    "lines.linewidth": 2.2,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


VIR = mpl.colormaps["viridis"]
MAG = mpl.colormaps["magma"]

BLACK = "#111111"
RED = "#b2182b"
GREY = "#666666"


def bbox(
    pad=0.20,
    lw=1.15,
):
    return dict(
        boxstyle=f"round,pad={pad}",
        fc="white",
        ec=RED,
        lw=lw,
        alpha=0.97,
    )


def clean(
    ax,
    axis="both",
):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis=axis,
        color=BLACK,
        linewidth=0.7,
        alpha=0.14,
    )

    ax.set_axisbelow(True)


def save(
    fig,
    stem,
):
    fig.savefig(
        FIG / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
    )

    fig.savefig(
        FIG / f"{stem}.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    plt.close(fig)

    print("WROTE:", stem)


# ============================================================
# 1. Parse DONE runtimes from logs
# ============================================================

print("\n[1] scanning logs")


log_paths = []

for root in [
    ROOT / "outputs",
    ROOT / "STATS",
    ROOT / "AgentUnlearn_full_code",
]:
    if not root.exists():
        continue

    for pattern in [
        "*.log",
        "*.txt",
        "*.out",
    ]:
        log_paths.extend(
            root.rglob(pattern)
        )


# Deduplicate.
log_paths = sorted(
    set(log_paths)
)


# Handles:
# DONE AU000001 51.6s
# DONE AU000001 51.6 s
# DONE PREFLIGHT01 8.6s
done_patterns = [
    re.compile(
        r"\bDONE\s+([A-Za-z0-9_.:-]+)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s*s\b"
    ),

    re.compile(
        r"\bDONE\s+([A-Za-z0-9_.:-]+).*?"
        r"([0-9]+(?:\.[0-9]+)?)\s*seconds?\b",
        re.I,
    ),
]


log_records = []


for path in log_paths:

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    for lineno, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        for pat in done_patterns:
            m = pat.search(
                line
            )

            if not m:
                continue

            log_records.append({
                "run_id":
                    m.group(1),

                "runtime_seconds_log":
                    float(
                        m.group(2)
                    ),

                "source_log":
                    str(path),

                "source_line":
                    lineno,
            })

            break


log_df = pd.DataFrame(
    log_records
)


if len(log_df):

    # If a run appears more than once, take median.
    log_summary = (
        log_df
        .groupby(
            "run_id"
        )
        .agg(
            runtime_seconds_log=(
                "runtime_seconds_log",
                "median",
            ),

            n_log_occurrences=(
                "runtime_seconds_log",
                "size",
            ),
        )
        .reset_index()
    )

else:

    log_summary = pd.DataFrame(
        columns=[
            "run_id",
            "runtime_seconds_log",
            "n_log_occurrences",
        ]
    )


log_df.to_csv(
    TECH / "runtime_log_hits.csv",
    index=False,
)


# ============================================================
# 2. Scan raw JSON timing fields
# ============================================================

print("\n[2] scanning raw JSON timing fields")


TIME_WORDS = (
    "time",
    "runtime",
    "duration",
    "elapsed",
    "latency",
    "wall",
    "second",
    "seconds",
)


EXCLUDE_WORDS = (
    "timestamp",
    "temperature",
    "threshold",
    "seed",
)


raw_long = []


def timing_leaves(
    obj,
    path="",
):

    out = []

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            p = (
                f"{path}.{key}"
                if path
                else str(key)
            )

            out.extend(
                timing_leaves(
                    value,
                    p,
                )
            )

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            # Normalize array indices away.
            p = (
                f"{path}[]"
            )

            out.extend(
                timing_leaves(
                    value,
                    p,
                )
            )

    elif isinstance(
        obj,
        (int, float),
    ) and not isinstance(
        obj,
        bool,
    ):

        low = path.lower()

        if (
            any(
                w in low
                for w in TIME_WORDS
            )
            and not any(
                w in low
                for w in EXCLUDE_WORDS
            )
        ):
            out.append(
                (
                    path,
                    float(obj),
                )
            )

    return out


for idx, path in enumerate(
    sorted(
        RAW.glob("*.json")
    ),
    start=1,
):

    try:

        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        continue

    run_id = path.stem

    leaves = timing_leaves(
        obj
    )

    grouped = defaultdict(
        list
    )

    for key, value in leaves:

        grouped[key].append(
            value
        )

    for key, values in grouped.items():

        values = np.asarray(
            values,
            dtype=float,
        )

        raw_long.append({
            "run_id":
                run_id,

            "timing_field":
                key,

            "n_values":
                len(values),

            "sum":
                values.sum(),

            "mean":
                values.mean(),

            "median":
                np.median(
                    values
                ),

            "max":
                values.max(),
        })


raw_timing = pd.DataFrame(
    raw_long
)


raw_timing.to_csv(
    TECH
    / "raw_timing_fields_long.csv",
    index=False,
)


# Timing field inventory.
if len(raw_timing):

    field_inventory = (
        raw_timing
        .groupby(
            "timing_field"
        )
        .agg(
            n_runs=(
                "run_id",
                "nunique",
            ),

            median_sum=(
                "sum",
                "median",
            ),

            mean_sum=(
                "sum",
                "mean",
            ),
        )
        .sort_values(
            "n_runs",
            ascending=False,
        )
        .reset_index()
    )

else:

    field_inventory = pd.DataFrame()


field_inventory.to_csv(
    TECH
    / "timing_fields_discovered.csv",
    index=False,
)


# ============================================================
# 3. Select a runtime source
# ============================================================

print("\n[3] choosing canonical runtime")


base = runs.copy()


if "run_id" not in base.columns:
    raise RuntimeError(
        "run_metrics_calibrated.csv has no run_id"
    )


base = base.merge(
    log_summary,
    on="run_id",
    how="left",
)


# Prefer actual runner DONE time.
base["runtime_seconds"] = (
    base[
        "runtime_seconds_log"
    ]
)


base["runtime_source"] = np.where(
    base[
        "runtime_seconds_log"
    ].notna(),
    "runner_DONE_log",
    np.nan,
)


# ------------------------------------------------------------
# Raw JSON fallback
# ------------------------------------------------------------

if len(raw_timing):

    candidates = (
        raw_timing
        .copy()
    )

    candidates[
        "_field_low"
    ] = candidates[
        "timing_field"
    ].str.lower()


    # Strong preference for explicit run-level totals.
    def score_field(
        field,
    ):

        f = field.lower()

        score = 0

        if "total" in f:
            score += 50

        if "elapsed" in f:
            score += 40

        if "runtime" in f:
            score += 40

        if "wall" in f:
            score += 30

        if "run" in f:
            score += 15

        if "generation" in f:
            score += 10

        if "probe" in f:
            score -= 5

        if "agent" in f:
            score -= 5

        return score


    candidates[
        "_priority"
    ] = candidates[
        "timing_field"
    ].map(
        score_field
    )


    best_fields = (
        candidates[
            [
                "timing_field",
                "_priority",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "_priority",
            ascending=False,
        )
    )


    if len(
        best_fields
    ):

        best_field = (
            best_fields
            .iloc[0][
                "timing_field"
            ]
        )

        fallback = (
            candidates[
                candidates[
                    "timing_field"
                ]
                == best_field
            ][
                [
                    "run_id",
                    "sum",
                ]
            ]
            .rename(
                columns={
                    "sum":
                        "runtime_raw_fallback"
                }
            )
        )

        base = base.merge(
            fallback,
            on="run_id",
            how="left",
        )


        missing = (
            base[
                "runtime_seconds"
            ].isna()
            &
            base[
                "runtime_raw_fallback"
            ].notna()
        )


        base.loc[
            missing,
            "runtime_seconds",
        ] = base.loc[
            missing,
            "runtime_raw_fallback",
        ]


        base.loc[
            missing,
            "runtime_source",
        ] = (
            "raw_json:"
            +
            best_field
        )

        print(
            "raw fallback field:",
            best_field,
        )


# ============================================================
# 4. Derived technical quantities
# ============================================================

print("\n[4] deriving technical metrics")


def infer_params_b(
    model_id,
):

    s = str(
        model_id
    ).lower()


    known = {
        "qwen3_4b": 4.0,
        "qwen3_8b": 8.0,
        "qwen3_14b": 14.0,

        "mistral_7b": 7.0,

        "qwen25coder_1p5b": 1.5,
        "qwen25coder_7b": 7.0,
        "qwen25coder_14b": 14.0,

        "qwen2.5_coder_1.5b": 1.5,
        "qwen2.5_coder_7b": 7.0,
        "qwen2.5_coder_14b": 14.0,
    }


    if s in known:
        return known[s]


    # 1p5b
    m = re.search(
        r"(\d+)p(\d+)b",
        s,
    )

    if m:
        return float(
            f"{m.group(1)}.{m.group(2)}"
        )


    # 14b / 7b / 4b ...
    m = re.search(
        r"(\d+(?:\.\d+)?)b",
        s,
    )

    if m:
        return float(
            m.group(1)
        )


    return np.nan


base[
    "model_params_b"
] = base[
    "model_id"
].map(
    infer_params_b
)


if "n_scored_probes" in base.columns:

    base[
        "seconds_per_probe"
    ] = (
        base[
            "runtime_seconds"
        ]
        /
        base[
            "n_scored_probes"
        ]
    )

else:

    base[
        "seconds_per_probe"
    ] = np.nan


base[
    "aggregate_run_hours"
] = (
    base[
        "runtime_seconds"
    ]
    / 3600
)


base.to_csv(
    TECH
    / "run_runtime_master.csv",
    index=False,
)


# ============================================================
# 5. Generic summaries
# ============================================================

print("\n[5] aggregating dimensions")


def summarize(
    cols,
    name,
):

    existing = [
        c
        for c in cols
        if c in base.columns
    ]


    d = base.dropna(
        subset=[
            "runtime_seconds"
        ]
    )


    if not existing:
        return


    summary = (
        d.groupby(
            existing,
            dropna=False,
        )
        .agg(
            n_runs=(
                "run_id",
                "size",
            ),

            total_hours=(
                "aggregate_run_hours",
                "sum",
            ),

            mean_seconds=(
                "runtime_seconds",
                "mean",
            ),

            median_seconds=(
                "runtime_seconds",
                "median",
            ),

            p90_seconds=(
                "runtime_seconds",
                lambda x:
                    np.quantile(
                        x,
                        0.90,
                    ),
            ),

            p95_seconds=(
                "runtime_seconds",
                lambda x:
                    np.quantile(
                        x,
                        0.95,
                    ),
            ),

            mean_sec_per_probe=(
                "seconds_per_probe",
                "mean",
            ),

            median_sec_per_probe=(
                "seconds_per_probe",
                "median",
            ),
        )
        .reset_index()
    )


    summary.to_csv(
        TECH / name,
        index=False,
    )


summarize(
    [
        "model_id",
        "model_params_b",
    ],
    "runtime_by_model.csv",
)


summarize(
    [
        "topology",
    ],
    "runtime_by_topology.csv",
)


summarize(
    [
        "intervention",
    ],
    "runtime_by_intervention.csv",
)


summarize(
    [
        "benchmark",
    ],
    "runtime_by_benchmark.csv",
)


summarize(
    [
        "block",
    ],
    "runtime_by_block.csv",
)


summarize(
    [
        "parameter_method",
    ],
    "runtime_by_parameter_method.csv",
)


summarize(
    [
        "model_id",
        "topology",
    ],
    "runtime_model_x_topology.csv",
)


summarize(
    [
        "model_id",
        "intervention",
    ],
    "runtime_model_x_intervention.csv",
)


summarize(
    [
        "topology",
        "intervention",
    ],
    "runtime_topology_x_intervention.csv",
)


# ============================================================
# 6. Paired intervention overhead
# ============================================================

print("\n[6] paired intervention overhead")


PAIR_KEYS = [
    c
    for c in [
        "block",
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
    if c in base.columns
]


pair = base.dropna(
    subset=[
        "runtime_seconds"
    ]
).copy()


for c in PAIR_KEYS:

    pair[c] = (
        pair[c]
        .astype(object)
        .where(
            pair[c].notna(),
            "__NA__",
        )
    )


runtime_pivot = (
    pair
    .pivot_table(
        index=PAIR_KEYS,
        columns="intervention",
        values="runtime_seconds",
        aggfunc="median",
    )
)


overhead_rows = []


CONTRASTS = [
    (
        "local_suppression",
        "none",
        "target_context_memory",
    ),

    (
        "communication_defense",
        "target_context_memory",
        "full_inference",
    ),

    (
        "total_defense",
        "none",
        "full_inference",
    ),
]


for contrast, A, B in CONTRASTS:

    if (
        A not in runtime_pivot.columns
        or B not in runtime_pivot.columns
    ):
        continue


    p = runtime_pivot[
        [A, B]
    ].dropna()


    if p.empty:
        continue


    delta = (
        p[B]
        -
        p[A]
    )


    ratio = (
        p[B]
        /
        p[A].replace(
            0,
            np.nan,
        )
    )


    overhead_rows.append({
        "contrast":
            contrast,

        "A":
            A,

        "B":
            B,

        "n_pairs":
            len(p),

        "mean_delta_seconds":
            delta.mean(),

        "median_delta_seconds":
            delta.median(),

        "mean_ratio":
            ratio.mean(),

        "median_ratio":
            ratio.median(),

        "p90_ratio":
            ratio.quantile(
                0.90
            ),
    })


overhead = pd.DataFrame(
    overhead_rows
)


overhead.to_csv(
    TECH
    / "paired_runtime_overhead.csv",
    index=False,
)


# ============================================================
# 7. Total workload summary
# ============================================================

valid = base.dropna(
    subset=[
        "runtime_seconds"
    ]
)


summary = {
    "runs_total":
        int(
            len(base)
        ),

    "runs_with_runtime":
        int(
            valid[
                "run_id"
            ].nunique()
        ),

    "coverage_fraction":
        float(
            valid[
                "run_id"
            ].nunique()
            /
            max(
                1,
                base[
                    "run_id"
                ].nunique(),
            )
        ),

    "aggregate_run_hours":
        float(
            valid[
                "runtime_seconds"
            ].sum()
            / 3600
        ),

    "aggregate_run_days":
        float(
            valid[
                "runtime_seconds"
            ].sum()
            / 86400
        ),

    "median_run_seconds":
        float(
            valid[
                "runtime_seconds"
            ].median()
        )
        if len(valid)
        else None,

    "p95_run_seconds":
        float(
            valid[
                "runtime_seconds"
            ].quantile(
                0.95
            )
        )
        if len(valid)
        else None,

    "runtime_sources":
        (
            valid[
                "runtime_source"
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        ),
}


(
    TECH
    / "RUNTIME_SUMMARY.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


# ============================================================
# Stop plots if runtime coverage is poor
# ============================================================

print()
print("=" * 90)
print("RUNTIME EXTRACTION")
print("=" * 90)
print(
    json.dumps(
        summary,
        indent=2,
    )
)


if (
    summary[
        "runs_with_runtime"
    ]
    < 10
):

    print(
        "\nToo few production runtimes for figures."
    )

    print(
        "Check TECH_STATS/timing_fields_discovered.csv "
        "and runtime_log_hits.csv"
    )

    raise SystemExit(0)


# ============================================================
# FIGURE 16 — runtime raincloud by model
# ============================================================

print("\n[7] FIGURE 16")


fig, ax = plt.subplots(
    figsize=(
        11.0,
        7.4,
    )
)


models = (
    valid.groupby(
        "model_id"
    )[
        "runtime_seconds"
    ]
    .median()
    .sort_values()
    .index
    .tolist()
)


datasets = [
    valid.loc[
        valid[
            "model_id"
        ] == m,
        "runtime_seconds",
    ].to_numpy()
    for m in models
]


positions = np.arange(
    len(models)
)


vp = ax.violinplot(
    datasets,
    positions=positions,
    widths=0.82,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)


for i, body in enumerate(
    vp["bodies"]
):

    body.set_facecolor(
        VIR(
            0.15
            +
            0.70
            * i
            /
            max(
                1,
                len(models) - 1,
            )
        )
    )

    body.set_edgecolor(
        BLACK
    )

    body.set_linewidth(
        1.0
    )

    body.set_alpha(
        0.36
    )


rng = np.random.default_rng(
    42
)


for i, (
    model,
    values,
) in enumerate(
    zip(
        models,
        datasets,
    )
):

    # Don't draw 1000 overlapping points.
    draw = values

    if len(draw) > 350:

        draw = rng.choice(
            draw,
            size=350,
            replace=False,
        )


    jitter = rng.normal(
        0,
        0.055,
        size=len(draw),
    )


    ax.scatter(
        np.full(
            len(draw),
            i,
        )
        + jitter,
        draw,
        s=18,
        c=[
            VIR(
                0.20
                +
                0.65
                * i
                /
                max(
                    1,
                    len(models) - 1,
                )
            )
        ],
        edgecolors=BLACK,
        linewidths=0.3,
        alpha=0.24,
        zorder=3,
    )


    med = np.median(
        values
    )


    p95 = np.quantile(
        values,
        0.95,
    )


    ax.scatter(
        [i],
        [med],
        s=135,
        marker="D",
        c="white",
        edgecolors=RED,
        linewidths=1.8,
        zorder=7,
    )


    ax.text(
        i,
        p95 * 1.08,
        (
            f"med {med:.1f}s\n"
            f"p95 {p95:.1f}s"
        ),
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        bbox=bbox(
            pad=0.14,
        ),
    )


ax.set_xticks(
    positions,
    [
        x.replace(
            "_",
            "\n",
        )
        for x in models
    ],
)


ax.set_ylabel(
    "Run time (seconds)"
)


ax.set_yscale(
    "log"
)


ax.set_title(
    "Runtime raincloud: model-dependent generation cost",
    fontweight="bold",
)


clean(
    ax,
    "y",
)


fig.tight_layout()


save(
    fig,
    "16_runtime_raincloud_by_model",
)


# ============================================================
# FIGURE 17 — empirical scaling law
# ============================================================

print("\n[8] FIGURE 17")


model_summary = (
    valid
    .dropna(
        subset=[
            "model_params_b",
            "seconds_per_probe",
        ]
    )
    .groupby(
        [
            "model_id",
            "model_params_b",
        ]
    )
    .agg(
        runtime_per_probe=(
            "seconds_per_probe",
            "median",
        ),

        p25=(
            "seconds_per_probe",
            lambda x:
                np.quantile(
                    x,
                    0.25,
                ),
        ),

        p75=(
            "seconds_per_probe",
            lambda x:
                np.quantile(
                    x,
                    0.75,
                ),
        ),

        n=(
            "run_id",
            "size",
        ),
    )
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(
        10.3,
        7.4,
    )
)


if len(
    model_summary
):

    cvals = np.linspace(
        0.15,
        0.9,
        len(
            model_summary
        ),
    )


    for (
        row,
        c,
    ) in zip(
        model_summary.itertuples(),
        cvals,
    ):

        ax.errorbar(
            row.model_params_b,
            row.runtime_per_probe,

            yerr=[
                [
                    row.runtime_per_probe
                    -
                    row.p25
                ],

                [
                    row.p75
                    -
                    row.runtime_per_probe
                ],
            ],

            fmt="o",
            markersize=13,

            markerfacecolor=
                VIR(c),

            markeredgecolor=
                BLACK,

            markeredgewidth=1.1,

            ecolor=BLACK,
            capsize=5,

            zorder=5,
        )


        ax.text(
            row.model_params_b,
            row.runtime_per_probe
            * 1.12,

            row.model_id.replace(
                "_",
                " ",
            ),

            ha="center",
            va="bottom",

            fontsize=13.5,
            fontweight="bold",

            bbox=bbox(
                pad=0.13,
            ),
        )


    # log-log empirical slope
    fit = model_summary[
        (
            model_summary[
                "model_params_b"
            ] > 0
        )
        &
        (
            model_summary[
                "runtime_per_probe"
            ] > 0
        )
    ]


    if len(
        fit
    ) >= 3:

        lx = np.log(
            fit[
                "model_params_b"
            ]
        )

        ly = np.log(
            fit[
                "runtime_per_probe"
            ]
        )


        slope, intercept = np.polyfit(
            lx,
            ly,
            1,
        )


        xx = np.geomspace(
            fit[
                "model_params_b"
            ].min(),
            fit[
                "model_params_b"
            ].max(),
            150,
        )


        yy = np.exp(
            intercept
        ) * xx ** slope


        ax.plot(
            xx,
            yy,
            color=BLACK,
            linestyle="--",
            linewidth=2.0,
            alpha=0.62,
        )


        ax.text(
            0.035,
            0.965,
            (
                "Empirical scaling\n"
                f"time/probe ∝ params^{slope:.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            bbox=bbox(),
        )


ax.set_xscale(
    "log"
)


ax.set_yscale(
    "log"
)


ax.set_xlabel(
    "Model size (billions of parameters)"
)


ax.set_ylabel(
    "Median seconds per scored probe"
)


ax.set_title(
    "Empirical compute scaling across model sizes",
    fontweight="bold",
)


clean(
    ax,
    "both",
)


fig.tight_layout()


save(
    fig,
    "17_runtime_scaling_law",
)


# ============================================================
# FIGURE 18 — topology × intervention cost matrix
# ============================================================

print("\n[9] FIGURE 18")


ti = (
    valid.groupby(
        [
            "topology",
            "intervention",
        ]
    )[
        "seconds_per_probe"
    ]
    .median()
    .reset_index()
)


topologies = [
    x
    for x in [
        "independent",
        "sequential",
        "debate",
        "scratchpad",
        "hierarchical",
    ]
    if x in set(
        ti[
            "topology"
        ]
    )
]


interventions = list(
    ti[
        "intervention"
    ].drop_duplicates()
)


# Normalize each cell against same topology / none.
baseline = (
    ti[
        ti[
            "intervention"
        ]
        == "none"
    ]
    .set_index(
        "topology"
    )[
        "seconds_per_probe"
    ]
)


matrix = np.full(
    (
        len(
            topologies
        ),
        len(
            interventions
        ),
    ),
    np.nan,
)


for i, topology in enumerate(
    topologies
):

    for j, intervention in enumerate(
        interventions
    ):

        z = ti[
            (
                ti[
                    "topology"
                ]
                == topology
            )
            &
            (
                ti[
                    "intervention"
                ]
                == intervention
            )
        ]


        if z.empty:
            continue


        value = z.iloc[0][
            "seconds_per_probe"
        ]


        if (
            topology
            in baseline.index
            and
            baseline.loc[
                topology
            ] > 0
        ):

            matrix[
                i,
                j,
            ] = (
                value
                /
                baseline.loc[
                    topology
                ]
            )


fig, ax = plt.subplots(
    figsize=(
        max(
            13,
            1.25
            * len(
                interventions
            ),
        ),
        7.1,
    )
)


finite = matrix[
    np.isfinite(
        matrix
    )
]


vmax = (
    max(
        1.2,
        np.quantile(
            finite,
            0.95,
        ),
    )
    if len(
        finite
    )
    else 2
)


im = ax.imshow(
    matrix,
    aspect="auto",
    cmap="magma",
    vmin=0.8,
    vmax=vmax,
    alpha=0.84,
)


ax.set_yticks(
    np.arange(
        len(
            topologies
        )
    ),
    [
        x.title()
        for x in topologies
    ],
)


ax.set_xticks(
    np.arange(
        len(
            interventions
        )
    ),
    [
        x.replace(
            "_",
            "\n",
        )
        for x in interventions
    ],
)


ax.set_xticks(
    np.arange(
        -0.5,
        len(
            interventions
        ),
        1,
    ),
    minor=True,
)


ax.set_yticks(
    np.arange(
        -0.5,
        len(
            topologies
        ),
        1,
    ),
    minor=True,
)


ax.grid(
    which="minor",
    color=BLACK,
    linewidth=0.75,
    alpha=0.70,
)


ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


for i in range(
    matrix.shape[0]
):

    for j in range(
        matrix.shape[1]
    ):

        v = matrix[
            i,
            j,
        ]


        if np.isfinite(
            v
        ):

            ax.text(
                j,
                i,
                f"{v:.2f}×",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                bbox=bbox(
                    pad=0.10,
                    lw=0.8,
                ),
            )


cb = fig.colorbar(
    im,
    ax=ax,
    pad=0.015,
)


cb.set_label(
    "Runtime multiplier vs no intervention",
    fontsize=17,
)


ax.set_title(
    "Computational cost fingerprint of topology × intervention",
    fontweight="bold",
)


fig.tight_layout()


save(
    fig,
    "18_topology_intervention_runtime_matrix",
)


# ============================================================
# FIGURE 19 — compute-effectiveness frontier
# ============================================================

print("\n[10] FIGURE 19")


main_valid = valid[
    valid[
        "block"
    ]
    .astype(str)
    .eq(
        "main"
    )
].copy()


# communication comparison by model x topology x benchmark
keys = [
    c
    for c in [
        "model_id",
        "benchmark",
        "topology",
        "unit_id",
        "target_id",
        "seed",
    ]
    if c in main_valid.columns
]


subset = main_valid[
    main_valid[
        "intervention"
    ].isin([
        "target_context_memory",
        "full_inference",
    ])
].copy()


metric_pivot = subset.pivot_table(
    index=keys,
    columns="intervention",
    values=[
        "runtime_seconds",
        "drr_target",
        "irr_target",
    ],
    aggfunc="mean",
)


rows = []


if len(
    metric_pivot
):

    for idx, row in metric_pivot.iterrows():

        try:

            rt_a = row[
                (
                    "runtime_seconds",
                    "target_context_memory",
                )
            ]

            rt_b = row[
                (
                    "runtime_seconds",
                    "full_inference",
                )
            ]


            drr_a = row[
                (
                    "drr_target",
                    "target_context_memory",
                )
            ]

            drr_b = row[
                (
                    "drr_target",
                    "full_inference",
                )
            ]


            irr_a = row[
                (
                    "irr_target",
                    "target_context_memory",
                )
            ]

            irr_b = row[
                (
                    "irr_target",
                    "full_inference",
                )
            ]


        except KeyError:

            continue


        vals = [
            rt_a,
            rt_b,
            drr_a,
            drr_b,
            irr_a,
            irr_b,
        ]


        if not all(
            np.isfinite(
                vals
            )
        ):
            continue


        overhead = (
            rt_b
            -
            rt_a
        )


        drr_gain = (
            drr_a
            -
            drr_b
        ) * 100


        irr_gain = (
            irr_a
            -
            irr_b
        ) * 100


        meta = dict(
            zip(
                keys,
                (
                    idx
                    if isinstance(
                        idx,
                        tuple,
                    )
                    else (
                        idx,
                    )
                ),
            )
        )


        rows.append({
            **meta,

            "runtime_overhead_seconds":
                overhead,

            "drr_gain_pp":
                drr_gain,

            "irr_gain_pp":
                irr_gain,
        })


frontier = pd.DataFrame(
    rows
)


frontier.to_csv(
    TECH
    / "communication_compute_effectiveness.csv",
    index=False,
)


fig, ax = plt.subplots(
    figsize=(
        15.8,
        9.6,
    )
)


if len(
    frontier
):

    # aggregate to avoid point explosion
    group_cols = [
        c
        for c in [
            "model_id",
            "benchmark",
            "topology",
        ]
        if c in frontier.columns
    ]


    ag = (
        frontier.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            runtime_overhead_seconds=(
                "runtime_overhead_seconds",
                "median",
            ),

            drr_gain_pp=(
                "drr_gain_pp",
                "mean",
            ),

            irr_gain_pp=(
                "irr_gain_pp",
                "mean",
            ),
        )
        .reset_index()
    )


    models = list(
        ag[
            "model_id"
        ].drop_duplicates()
    )


    cmap_positions = {
        m:
            VIR(
                0.12
                +
                0.78
                * i
                /
                max(
                    1,
                    len(models) - 1,
                )
            )

        for i, m in enumerate(
            models
        )
    }


    for row in ag.itertuples():

        x = (
            row.runtime_overhead_seconds
        )

        y = (
            row.drr_gain_pp
        )

        size = (
            90
            +
            28
            * max(
                0,
                row.irr_gain_pp,
            )
        )


        ax.scatter(
            x,
            y,
            s=size,
            c=[
                cmap_positions[
                    row.model_id
                ]
            ],
            edgecolors=BLACK,
            linewidths=0.9,
            alpha=0.62,
            zorder=3,
        )


    # Pareto-ish labels: best gain / modest cost
    score = (
        ag[
            "drr_gain_pp"
        ]
        /
        (
            1
            +
            np.maximum(
                0,
                ag[
                    "runtime_overhead_seconds"
                ],
            )
        )
    )


    label_idx = score.nlargest(
        min(
            8,
            len(
                score
            ),
        )
    ).index


    # --------------------------------------------------------
    # Clean short callouts for a small set of key points.
    # Use offset-points coordinates so arrows always start from
    # the correct point instead of some distant data location.
    # --------------------------------------------------------

    labelled = ag.loc[label_idx].copy()

    if len(labelled):

        labelled = labelled.sort_values(
            "drr_gain_pp",
            ascending=False,
        ).copy()

        # Manual short offsets (dx, dy) in display points.
        # Chosen to keep boxes readable and non-overlapping
        # for the dense left cluster.
        offset_cycle = [
            (20, 18),
            (18, -18),
            (24, 34),
            (34, 8),
            (-18, 20),
            (-22, -14),
            (30, -24),
            (16, 48),
        ]

        for k, (_, row) in enumerate(labelled.iterrows()):

            dx, dy = offset_cycle[k % len(offset_cycle)]

            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"

            ax.annotate(
                (
                    f"{row['model_id']}\n"
                    f"{row['benchmark']} / {row['topology']}"
                ),
                xy=(
                    float(row["runtime_overhead_seconds"]),
                    float(row["drr_gain_pp"]),
                ),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=11.2,
                fontweight="bold",
                bbox=bbox(
                    pad=0.14,
                    lw=0.85,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color=BLACK,
                    linewidth=0.7,
                    alpha=0.65,
                    shrinkA=4,
                    shrinkB=4,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=9,
            )


    # Quadrants
    ax.axvline(
        0,
        color=BLACK,
        linestyle="--",
        linewidth=1,
    )


    ax.axhline(
        0,
        color=BLACK,
        linestyle="--",
        linewidth=1,
    )


    ax.text(
        0.98,
        0.98,
        "↑ more forgetting benefit\n← lower compute overhead",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=GREY,
        bbox=bbox(),
    )


ax.set_xlabel(
    "Additional runtime of communication filtering (s)"
)

# Focus on the practically relevant overhead range.
ax.set_xlim(0, 20)

# Slightly more top headroom for local callouts.
if len(frontier):
    _ymin = float(
        min(
            -12.0,
            frontier["drr_gain_pp"].min() - 6.0,
        )
    )
    _ymax = float(
        max(
            115.0,
            frontier["drr_gain_pp"].max() + 18.0,
        )
    )
    ax.set_ylim(_ymin, _ymax)


ax.set_ylabel(
    "Additional target-DRR suppression (pp)"
)


ax.set_title(
    "Compute-effectiveness frontier of communication defense",
    fontweight="bold",
    pad=22,
)


clean(
    ax,
    "both",
)

# Much more physical canvas around the axes.
fig.subplots_adjust(
    left=0.095,
    right=0.98,
    bottom=0.14,
    top=0.90,
)


save(
    fig,
    "19_compute_effectiveness_frontier",
)


# ============================================================
# Training timing scan
# ============================================================

print("\n[11] training timing")


training_summary = (
    ROOT
    / "STATS"
    / "06_training"
    / "training_summary.csv"
)


if training_summary.exists():

    tr = pd.read_csv(
        training_summary
    )


    timing_cols = [
        c
        for c in tr.columns
        if any(
            token in c.lower()
            for token in [
                "time",
                "second",
                "duration",
                "elapsed",
                "runtime",
            ]
        )
    ]


    if timing_cols:

        tr.to_csv(
            TECH
            / "training_timing_summary_full.csv",
            index=False,
        )


        (
            TECH
            / "training_timing_columns.txt"
        ).write_text(
            "\n".join(
                timing_cols
            )
        )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 94)
print("TECHNICAL RUNTIME ATLAS COMPLETE")
print("=" * 94)

print()
print("TECH_STATS:")
for p in sorted(
    TECH.iterdir()
):
    print(" ", p.name)

print()
print("FIGURES:")
for stem in [
    "16_runtime_raincloud_by_model",
    "17_runtime_scaling_law",
    "18_topology_intervention_runtime_matrix",
    "19_compute_effectiveness_frontier",
]:
    print(
        " ",
        stem,
    )

