#!/usr/bin/env python3

from pathlib import Path
import json
import math
import zlib

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

OUT = ROOT / "FIGURES"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Inputs
# ============================================================

runs = pd.read_csv(
    AUTO / "run_metrics_calibrated.csv"
)

primary = pd.read_csv(
    AUTO / "PRIMARY_18_CALIBRATED.csv"
)

agreement = pd.read_csv(
    AUTO / "inter_judge_agreement.csv"
)

sensitivity = pd.read_csv(
    AUTO / "THRESHOLD_SENSITIVITY.csv"
)

thresholds = json.loads(
    (
        AUTO
        / "calibrated_thresholds.json"
    ).read_text()
)


# ============================================================
# Global style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    # Intentionally oversized:
    # survives aggressive shrinking in paper layout.
    "font.size": 19,

    "axes.titlesize": 25,
    "axes.labelsize": 21,

    "xtick.labelsize": 17,
    "ytick.labelsize": 17,

    "legend.fontsize": 15,

    "axes.linewidth": 1.15,
    "lines.linewidth": 2.4,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


VIR = mpl.colormaps["viridis"]

RED = "#b2182b"
BLACK = "#111111"
GREY = "#5f5f5f"


def callout_box(
    pad=0.26,
    lw=1.35,
):
    return dict(
        boxstyle=f"round,pad={pad}",
        fc="white",
        ec=RED,
        lw=lw,
        alpha=0.97,
    )


def save(
    fig,
    stem,
):
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    fig.savefig(
        png,
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    plt.close(fig)

    print("WROTE:", pdf)
    print("WROTE:", png)


def beautify(
    ax,
    grid_axis="y",
):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis=grid_axis,
        color=BLACK,
        linewidth=0.75,
        alpha=0.15,
    )

    ax.set_axisbelow(True)


def fmt_p(
    p,
):
    if not np.isfinite(p):
        return "n/a"

    if p < 1e-4:
        return f"{p:.1e}"

    if p < 1e-3:
        return f"{p:.4f}"

    return f"{p:.3f}"


# ============================================================
# Metrics
# ============================================================

METRICS = [
    "drr_target",
    "irr_target",
    "retention_target",

    "drr_system",
    "irr_system",
    "retention_system",
]


LABELS = {
    "drr_target": "Target DRR",
    "irr_target": "Target IRR",
    "retention_target": "Target retention",

    "drr_system": "System DRR",
    "irr_system": "System IRR",
    "retention_system": "System retention",
}


SHORT = {
    "drr_target": "T-DRR",
    "irr_target": "T-IRR",
    "retention_target": "T-Ret",

    "drr_system": "S-DRR",
    "irr_system": "S-IRR",
    "retention_system": "S-Ret",
}


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


# ============================================================
# Utilities
# ============================================================

def stable_seed(
    *parts,
):
    s = "|".join(
        map(str, parts)
    )

    return (
        zlib.crc32(
            s.encode("utf-8")
        )
        & 0xffffffff
    )


def bootstrap_ci(
    x,
    n_boot=5000,
    seed=42,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return (
            float(x[0]),
            float(x[0]),
        )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(x),
        size=(
            n_boot,
            len(x),
        ),
    )

    means = x[idx].mean(
        axis=1
    )

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def gradient_barh(
    ax,
    y,
    value,
    height=0.60,
    cmap="viridis",
    alpha=0.80,
):
    value = float(value)

    lo = min(
        0.0,
        value,
    )

    hi = max(
        0.0,
        value,
    )

    if math.isclose(
        lo,
        hi,
    ):
        return

    grad = np.linspace(
        0.10,
        0.96,
        400,
    ).reshape(
        1,
        -1,
    )

    if value < 0:
        grad = grad[
            :,
            ::-1,
        ]

    ax.imshow(
        grad,
        extent=(
            lo,
            hi,
            y - height / 2,
            y + height / 2,
        ),
        aspect="auto",
        cmap=cmap,
        interpolation="bicubic",
        alpha=alpha,
        zorder=2,
    )

    ax.add_patch(
        Rectangle(
            (
                lo,
                y - height / 2,
            ),
            hi - lo,
            height,
            fill=False,
            edgecolor=BLACK,
            linewidth=1.05,
            zorder=3,
        )
    )


def communication_effects_by(
    dimension,
):
    main = runs[
        runs["block"]
        .astype(str)
        .eq("main")
    ].copy()

    main = main[
        main["intervention"].isin([
            "target_context_memory",
            "full_inference",
        ])
    ].copy()

    rows = []

    for value, group in main.groupby(
        dimension,
        dropna=False,
    ):
        group = group.copy()

        keys = [
            c
            for c in PAIR_KEYS
            if c in group.columns
        ]

        for c in keys:
            group[c] = (
                group[c]
                .astype(object)
                .where(
                    group[c].notna(),
                    "__NA__",
                )
            )

        for metric in METRICS:
            t = group[
                keys
                + [
                    "intervention",
                    metric,
                ]
            ].dropna(
                subset=[metric]
            )

            pivot = t.pivot_table(
                index=keys,
                columns="intervention",
                values=metric,
                aggfunc="mean",
                dropna=False,
            )

            needed = {
                "target_context_memory",
                "full_inference",
            }

            if not needed.issubset(
                set(pivot.columns)
            ):
                continue

            pivot = pivot[
                [
                    "target_context_memory",
                    "full_inference",
                ]
            ].dropna()

            if pivot.empty:
                continue

            diff = (
                pivot["full_inference"]
                -
                pivot[
                    "target_context_memory"
                ]
            ).to_numpy(
                dtype=float
            )

            lo, hi = bootstrap_ci(
                diff,
                seed=stable_seed(
                    dimension,
                    value,
                    metric,
                ),
            )

            rows.append({
                dimension: value,
                "metric": metric,
                "delta": diff.mean(),
                "lo": lo,
                "hi": hi,
                "n": len(diff),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# FIGURE 1
# Causal trajectory
# ============================================================

print("\n[1/7] causal trajectory")

main = runs[
    runs["block"]
    .astype(str)
    .eq("main")
].copy()


INTERVENTIONS = [
    "none",
    "target_context_memory",
    "full_inference",
]


STAGE_LABELS = [
    "Baseline",
    "Local\nsuppression",
    "+ Communication\nfiltering",
]


means = (
    main[
        main["intervention"]
        .isin(INTERVENTIONS)
    ]
    .groupby("intervention")[
        METRICS
    ]
    .mean()
    .reindex(
        INTERVENTIONS
    )
    * 100
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.2, 6.4),
    sharey=True,
)


panels = [
    (
        axes[0],
        "Target agent",
        [
            "drr_target",
            "irr_target",
            "retention_target",
        ],
    ),
    (
        axes[1],
        "Whole system",
        [
            "drr_system",
            "irr_system",
            "retention_system",
        ],
    ),
]


colors = [
    VIR(0.20),
    VIR(0.52),
    VIR(0.84),
]


for ax, title, metrics in panels:
    x = np.arange(
        len(INTERVENTIONS)
    )

    # very subtle viridis stage curtains
    for xi, cpos in zip(
        x,
        [0.15, 0.50, 0.84],
    ):
        ax.axvspan(
            xi - 0.23,
            xi + 0.23,
            color=VIR(cpos),
            alpha=0.05,
            zorder=0,
        )

    for metric, color in zip(
        metrics,
        colors,
    ):
        y = means[
            metric
        ].to_numpy(
            dtype=float
        )

        ax.fill_between(
            x,
            y,
            0,
            color=color,
            alpha=0.055,
            zorder=1,
        )

        ax.plot(
            x,
            y,
            marker="o",
            markersize=10,
            markeredgecolor=BLACK,
            markeredgewidth=1.2,
            color=color,
            zorder=4,
            label=(
                LABELS[metric]
                .replace(
                    "Target ",
                    "",
                )
                .replace(
                    "System ",
                    "",
                )
            ),
        )

        for xi, yi in zip(
            x,
            y,
        ):
            yoff = (
                -5.0
                if yi > 92
                else 2.2
            )

            ax.text(
                xi,
                yi + yoff,
                f"{yi:.1f}%",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                bbox=callout_box(),
                zorder=10,
            )

    ax.set_xticks(
        x,
        STAGE_LABELS,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=10,
    )

    beautify(
        ax,
        "y",
    )

    ax.legend(
        frameon=False,
        loc="lower left",
    )


axes[0].set_ylabel(
    "Rate (%)"
)


fig.suptitle(
    "Causal trajectory of agent-level unlearning",
    fontsize=27,
    fontweight="bold",
    y=1.02,
)


fig.text(
    0.5,
    -0.012,
    "Lower DRR/IRR is better; higher retention is better.",
    ha="center",
    fontsize=17,
    color=GREY,
)


fig.tight_layout()

save(
    fig,
    "01_causal_trajectory",
)


# ============================================================
# FIGURE 2
# Confirmatory communication forest
# ============================================================

print("\n[2/7] communication forest")

d = primary[
    primary["contrast"]
    .eq("communication_defense")
].copy()


d["delta_pp"] = (
    d[
        "mean_diff_B_minus_A"
    ]
    * 100
)

d["lo_pp"] = (
    d["bootstrap95_low"]
    * 100
)

d["hi_pp"] = (
    d["bootstrap95_high"]
    * 100
)


d = (
    d
    .set_index("metric")
    .reindex(METRICS)
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(9.8, 7.2)
)


ys = np.arange(
    len(d)
)[::-1]


for y, row in zip(
    ys,
    d.itertuples(),
):
    gradient_barh(
        ax,
        y,
        row.delta_pp,
        height=0.56,
        alpha=0.80,
    )

    ax.errorbar(
        row.delta_pp,
        y,
        xerr=[
            [
                row.delta_pp
                -
                row.lo_pp
            ],
            [
                row.hi_pp
                -
                row.delta_pp
            ],
        ],
        fmt="o",
        markersize=8,
        markerfacecolor=VIR(0.72),
        markeredgecolor=BLACK,
        markeredgewidth=1.15,
        ecolor=BLACK,
        elinewidth=1.3,
        capsize=4,
        zorder=5,
    )

    left = (
        row.delta_pp < 0
    )

    ax.text(
        row.delta_pp
        + (
            -0.7
            if left
            else 0.7
        ),
        y,
        (
            f"{row.delta_pp:+.1f} pp\n"
            f"Holm p={fmt_p(row.wilcoxon_p_holm18)}"
        ),
        ha=(
            "right"
            if left
            else "left"
        ),
        va="center",
        fontsize=13.2,
        fontweight="bold",
        bbox=callout_box(
            pad=0.20,
        ),
        zorder=9,
    )


ax.axvline(
    0,
    color=BLACK,
    linewidth=1.1,
    linestyle="--",
)


ax.axhline(
    2.5,
    color=BLACK,
    linewidth=0.8,
    alpha=0.45,
)


ax.set_yticks(
    ys,
    [
        LABELS[m]
        for m in d["metric"]
    ],
)


ax.set_xlabel(
    "Effect of adding communication filtering (percentage points)"
)


ax.set_title(
    "Communication defense: selective target suppression",
    fontweight="bold",
)


beautify(
    ax,
    "x",
)


fig.tight_layout()

save(
    fig,
    "02_communication_defense_forest",
)


# ============================================================
# Heatmap factory
# ============================================================

def effect_matrix_figure(
    dimension,
    stem,
    title,
    preferred_order=None,
):
    eff = communication_effects_by(
        dimension
    )

    if eff.empty:
        print(
            "SKIP empty:",
            stem,
        )
        return

    values = list(
        eff[
            dimension
        ].drop_duplicates()
    )

    if preferred_order:
        present = set(
            values
        )

        ordered = [
            x
            for x in preferred_order
            if x in present
        ]

        ordered += [
            x
            for x in values
            if x not in ordered
        ]

        values = ordered

    matrix = np.full(
        (
            len(values),
            len(METRICS),
        ),
        np.nan,
    )

    for i, v in enumerate(
        values
    ):
        for j, metric in enumerate(
            METRICS
        ):
            z = eff[
                (
                    eff[dimension]
                    == v
                )
                &
                (
                    eff["metric"]
                    == metric
                )
            ]

            if len(z):
                matrix[
                    i,
                    j,
                ] = (
                    100
                    * z.iloc[0][
                        "delta"
                    ]
                )

    finite = matrix[
        np.isfinite(matrix)
    ]

    lim = (
        max(
            1.0,
            float(
                np.max(
                    np.abs(
                        finite
                    )
                )
            ),
        )
        if len(finite)
        else 1.0
    )

    norm = TwoSlopeNorm(
        vmin=-lim,
        vcenter=0,
        vmax=lim,
    )

    height = max(
        5.1,
        1.45
        + 0.82
        * len(values),
    )

    fig, ax = plt.subplots(
        figsize=(
            11.4,
            height,
        )
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        alpha=0.88,
    )

    ax.set_xticks(
        np.arange(
            len(METRICS)
        ),
        [
            SHORT[m]
            for m in METRICS
        ],
    )

    ax.set_yticks(
        np.arange(
            len(values)
        ),
        [
            str(x)
            .replace("_", " ")
            .title()
            for x in values
        ],
    )

    # Thin black cell borders
    ax.set_xticks(
        np.arange(
            -0.5,
            len(METRICS),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(values),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color=BLACK,
        linewidth=0.72,
        alpha=0.78,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for i in range(
        len(values)
    ):
        for j in range(
            len(METRICS)
        ):
            value = matrix[
                i,
                j,
            ]

            if np.isfinite(
                value
            ):
                ax.text(
                    j,
                    i,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=14.2,
                    fontweight="bold",
                    bbox=callout_box(
                        pad=0.11,
                        lw=0.85,
                    ),
                )

            else:
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=19,
                    color=GREY,
                )

    cb = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        fraction=0.05,
    )

    cb.set_label(
        "Communication-defense effect (pp)",
        fontsize=17,
    )

    cb.ax.tick_params(
        labelsize=14,
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Negative DRR/IRR = stronger suppression; retention ≈ 0 = preserved utility",
        fontsize=15.5,
    )

    fig.tight_layout()

    save(
        fig,
        stem,
    )


# ============================================================
# FIGURE 3
# Topology
# ============================================================

print("\n[3/7] topology matrix")

effect_matrix_figure(
    "topology",
    "03_topology_signature",
    "Topology fingerprint of communication-mediated reconstruction",
    [
        "independent",
        "sequential",
        "debate",
        "scratchpad",
        "hierarchical",
    ],
)


# ============================================================
# FIGURE 4
# Benchmark
# ============================================================

print("\n[4/7] benchmark matrix")

effect_matrix_figure(
    "benchmark",
    "04_benchmark_signature",
    "Benchmark fingerprint of communication defense",
    [
        "TOFU",
        "RWKU",
        "MUSE-Books",
        "MUSE-News",
        "WMDP",
    ],
)


# ============================================================
# FIGURE 5
# Model + parameter-level
# ============================================================

print("\n[5/7] model + parameter landscape")

model_eff = communication_effects_by(
    "model_id"
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.6, 6.5),
)


# -----------------------
# Panel A: model effects
# -----------------------

ax = axes[0]


preferred_models = [
    "qwen3_4b",
    "qwen3_8b",
    "mistral_7b",
]


available_models = set(
    model_eff[
        "model_id"
    ].dropna()
)


models = [
    m
    for m in preferred_models
    if m in available_models
]


models += sorted([
    m
    for m in available_models
    if m not in models
])


ybase = np.arange(
    len(models)
)[::-1]


for metric_index, metric in enumerate([
    "drr_target",
    "irr_target",
]):
    offset = (
        0.16
        if metric_index == 0
        else -0.16
    )

    color = (
        VIR(0.26)
        if metric_index == 0
        else VIR(0.76)
    )

    for ii, (
        y,
        model,
    ) in enumerate(
        zip(
            ybase,
            models,
        )
    ):
        z = model_eff[
            (
                model_eff["model_id"]
                == model
            )
            &
            (
                model_eff["metric"]
                == metric
            )
        ]

        if z.empty:
            continue

        row = z.iloc[0]

        value = (
            100
            * row["delta"]
        )

        lo = (
            100
            * row["lo"]
        )

        hi = (
            100
            * row["hi"]
        )

        ax.errorbar(
            value,
            y + offset,
            xerr=[
                [value - lo],
                [hi - value],
            ],
            fmt="o",
            markersize=9,
            markerfacecolor=color,
            markeredgecolor=BLACK,
            markeredgewidth=1.0,
            ecolor=BLACK,
            capsize=4,
            label=(
                LABELS[metric]
                if ii == 0
                else None
            ),
        )

        ax.text(
            value
            + (
                -0.48
                if value < 0
                else 0.48
            ),
            y + offset,
            f"{value:+.1f}",
            ha=(
                "right"
                if value < 0
                else "left"
            ),
            va="center",
            fontsize=12.7,
            fontweight="bold",
            bbox=callout_box(
                pad=0.12,
                lw=0.8,
            ),
        )


ax.axvline(
    0,
    color=BLACK,
    linewidth=1,
    linestyle="--",
)


ax.set_yticks(
    ybase,
    [
        m.replace(
            "_",
            " ",
        )
        for m in models
    ],
)


ax.set_xlabel(
    "Communication-defense effect (pp)"
)


ax.set_title(
    "Model-family consistency",
    fontweight="bold",
)


ax.legend(
    frameon=False,
)


beautify(
    ax,
    "x",
)


# -----------------------
# Panel B: parametric trade-off
# -----------------------

ax = axes[1]


block = runs[
    "block"
].astype(str)


parameter_method = (
    runs[
        "parameter_method"
    ]
    .astype(str)
    .str.lower()
)


parameter_runs = runs[
    (
        block.str.contains(
            "parameter",
            case=False,
            na=False,
        )
    )
    |
    (
        ~parameter_method.isin([
            "none",
            "nan",
            "",
        ])
    )
].copy()


if len(parameter_runs):
    g = (
        parameter_runs
        .groupby(
            "parameter_method",
            dropna=False,
        )[
            [
                "drr_target",
                "retention_target",
            ]
        ]
        .mean()
        .dropna(
            how="all"
        )
    )

    colors = VIR(
        np.linspace(
            0.20,
            0.88,
            max(
                1,
                len(g),
            ),
        )
    )

    for (
        method,
        row,
    ), color in zip(
        g.iterrows(),
        colors,
    ):
        x = (
            100
            * row[
                "drr_target"
            ]
        )

        y = (
            100
            * row[
                "retention_target"
            ]
        )

        ax.scatter(
            x,
            y,
            s=330,
            c=[color],
            edgecolors=BLACK,
            linewidths=1.35,
            alpha=0.86,
            zorder=4,
        )

        ax.text(
            x,
            y + 2.0,
            str(method)
            .replace(
                "_",
                " ",
            ),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            bbox=callout_box(),
            zorder=7,
        )

    ax.set_xlabel(
        "Target DRR (%)  ← lower is better"
    )

    ax.set_ylabel(
        "Target retention (%)  ↑ higher is better"
    )

    # Decorative improvement directions
    ax.text(
        0.03,
        0.06,
        "← stronger forgetting",
        transform=ax.transAxes,
        fontsize=14,
        color=GREY,
        fontweight="bold",
    )

    ax.text(
        0.61,
        0.94,
        "utility ↑",
        transform=ax.transAxes,
        fontsize=14,
        color=GREY,
        fontweight="bold",
    )

else:
    ax.text(
        0.5,
        0.5,
        "No parameter-level rows",
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox=callout_box(),
    )


ax.set_title(
    "Parameter-unlearning trade-off",
    fontweight="bold",
)


beautify(
    ax,
    "both",
)


fig.suptitle(
    "Model and parameter-level robustness",
    fontsize=26,
    fontweight="bold",
    y=1.02,
)


fig.tight_layout()

save(
    fig,
    "05_model_and_parameter_landscape",
)


# ============================================================
# FIGURE 6
# Robustness + WMDP
# ============================================================

print("\n[6/7] robustness + WMDP")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.6, 6.4),
)


# -----------------------
# Panel A: seeds/temp
# -----------------------

ax = axes[0]


robust = runs[
    runs["block"]
    .astype(str)
    .str.contains(
        "robust",
        case=False,
        na=False,
    )
].copy()


robust_rows = []


if len(robust):
    robust = robust[
        robust[
            "intervention"
        ].isin([
            "target_context_memory",
            "full_inference",
        ])
    ].copy()

    for (
        temperature,
        seed,
    ), group in robust.groupby(
        [
            "temperature",
            "seed",
        ],
        dropna=False,
    ):
        group = group.copy()

        keys = [
            c
            for c in PAIR_KEYS
            if (
                c in group.columns
                and c not in [
                    "temperature",
                    "seed",
                ]
            )
        ]

        for c in keys:
            group[c] = (
                group[c]
                .astype(object)
                .where(
                    group[c].notna(),
                    "__NA__",
                )
            )

        for metric in [
            "drr_target",
            "irr_target",
        ]:
            t = group[
                keys
                + [
                    "intervention",
                    metric,
                ]
            ].dropna(
                subset=[metric]
            )

            pivot = t.pivot_table(
                index=keys,
                columns="intervention",
                values=metric,
                aggfunc="mean",
            )

            if not {
                "target_context_memory",
                "full_inference",
            }.issubset(
                set(
                    pivot.columns
                )
            ):
                continue

            pivot = pivot[
                [
                    "target_context_memory",
                    "full_inference",
                ]
            ].dropna()

            if pivot.empty:
                continue

            delta = (
                pivot[
                    "full_inference"
                ]
                -
                pivot[
                    "target_context_memory"
                ]
            ).mean()

            robust_rows.append({
                "temperature": temperature,
                "seed": seed,
                "metric": metric,
                "delta": (
                    100
                    * delta
                ),
            })


robust_effects = pd.DataFrame(
    robust_rows
)


if len(robust_effects):
    seeds = list(
        robust_effects[
            "seed"
        ].drop_duplicates()
    )

    for seed_index, seed in enumerate(
        seeds
    ):
        color = VIR(
            0.18
            +
            0.68
            * (
                seed_index
                /
                max(
                    1,
                    len(seeds) - 1,
                )
            )
        )

        for metric in [
            "drr_target",
            "irr_target",
        ]:
            z = robust_effects[
                (
                    robust_effects[
                        "seed"
                    ]
                    == seed
                )
                &
                (
                    robust_effects[
                        "metric"
                    ]
                    == metric
                )
            ].sort_values(
                "temperature"
            )

            if z.empty:
                continue

            ax.plot(
                z["temperature"],
                z["delta"],
                marker=(
                    "o"
                    if metric
                    == "drr_target"
                    else "s"
                ),
                markersize=8,
                markeredgecolor=BLACK,
                markeredgewidth=0.9,
                color=color,
                alpha=0.79,
                linestyle=(
                    "-"
                    if metric
                    == "drr_target"
                    else "--"
                ),
                label=(
                    f"{SHORT[metric]}, "
                    f"seed {seed}"
                ),
            )

    ax.axhline(
        0,
        color=BLACK,
        linewidth=1,
        linestyle="--",
    )

    ax.set_xlabel(
        "Generation temperature"
    )

    ax.set_ylabel(
        "Communication-defense effect (pp)"
    )

    ax.legend(
        frameon=False,
        fontsize=11.5,
        ncol=1,
    )

else:
    ax.text(
        0.5,
        0.5,
        "No robustness rows",
        transform=ax.transAxes,
        ha="center",
        bbox=callout_box(),
    )


ax.set_title(
    "Seed × temperature stability",
    fontweight="bold",
)


beautify(
    ax,
    "y",
)


# -----------------------
# Panel B: WMDP
# -----------------------

ax = axes[1]


wmdp = runs[
    runs["benchmark"]
    .astype(str)
    .str.upper()
    .eq("WMDP")
].copy()


if len(wmdp):
    available = [
        x
        for x in INTERVENTIONS
        if x in set(
            wmdp[
                "intervention"
            ]
        )
    ]

    g = (
        wmdp[
            wmdp[
                "intervention"
            ].isin(
                available
            )
        ]
        .groupby(
            "intervention"
        )[
            [
                "drr_target",
                "drr_system",
            ]
        ]
        .mean()
        .reindex(
            available
        )
        * 100
    )

    xpos = np.arange(
        len(g)
    )

    width = 0.32

    metric_specs = [
        (
            "drr_target",
            -width / 1.7,
            0.12,
            0.52,
        ),
        (
            "drr_system",
            width / 1.7,
            0.52,
            0.94,
        ),
    ]

    for metric, offset, c0, c1 in metric_specs:
        values = g[
            metric
        ].to_numpy(
            dtype=float
        )

        for i, value in enumerate(
            values
        ):
            center = (
                xpos[i]
                + offset
            )

            left = (
                center
                -
                width * 0.44
            )

            right = (
                center
                +
                width * 0.44
            )

            grad = np.linspace(
                c0,
                c1,
                280,
            ).reshape(
                -1,
                1,
            )

            ax.imshow(
                grad,
                extent=(
                    left,
                    right,
                    0,
                    value,
                ),
                aspect="auto",
                cmap="viridis",
                alpha=0.82,
                zorder=2,
            )

            ax.add_patch(
                Rectangle(
                    (
                        left,
                        0,
                    ),
                    right - left,
                    value,
                    fill=False,
                    edgecolor=BLACK,
                    linewidth=1.0,
                    zorder=3,
                )
            )

            ax.text(
                center,
                value + 2.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                bbox=callout_box(
                    pad=0.12,
                    lw=0.8,
                ),
                zorder=6,
            )

    ax.set_xticks(
        xpos,
        [
            x
            .replace(
                "_",
                " ",
            )
            for x in g.index
        ],
        rotation=10,
    )

    ymax = float(
        np.nanmax(
            g.to_numpy()
        )
    )

    ax.set_ylim(
        0,
        max(
            100,
            ymax + 12,
        ),
    )

    ax.set_ylabel(
        "Direct recall rate (%)"
    )

    ax.scatter(
        [],
        [],
        s=120,
        color=VIR(0.32),
        edgecolor=BLACK,
        label="Target DRR",
    )

    ax.scatter(
        [],
        [],
        s=120,
        color=VIR(0.78),
        edgecolor=BLACK,
        label="System DRR",
    )

    ax.legend(
        frameon=False,
    )

else:
    ax.text(
        0.5,
        0.5,
        "No WMDP rows",
        transform=ax.transAxes,
        ha="center",
        bbox=callout_box(),
    )


ax.set_title(
    "WMDP appendix stress test",
    fontweight="bold",
)


beautify(
    ax,
    "y",
)


fig.suptitle(
    "Robustness beyond the main factorial",
    fontsize=26,
    fontweight="bold",
    y=1.02,
)


fig.tight_layout()

save(
    fig,
    "06_robustness_and_wmdp",
)


# ============================================================
# FIGURE 7
# Scorer + threshold robustness atlas
# ============================================================

print("\n[7/7] threshold robustness atlas")


wide = sensitivity.pivot_table(
    index=[
        "content_f1_threshold",
        "semantic_threshold",
    ],
    columns="metric",
    values="communication_delta",
    aggfunc="mean",
).reset_index()


wide = wide.dropna(
    subset=[
        "drr_target",
        "irr_target",
    ]
)


x = (
    100
    * wide[
        "drr_target"
    ].to_numpy(
        dtype=float
    )
)


y = (
    100
    * wide[
        "irr_target"
    ].to_numpy(
        dtype=float
    )
)


semantic_t = wide[
    "semantic_threshold"
].to_numpy(
    dtype=float
)


content_t = wide[
    "content_f1_threshold"
].to_numpy(
    dtype=float
)


den = max(
    1e-9,
    content_t.max()
    -
    content_t.min(),
)


sizes = (
    70
    +
    230
    * (
        content_t
        -
        content_t.min()
    )
    /
    den
)


fig, ax = plt.subplots(
    figsize=(9.6, 7.4)
)


sc = ax.scatter(
    x,
    y,
    c=semantic_t,
    s=sizes,
    cmap="viridis",
    alpha=0.68,
    edgecolors=BLACK,
    linewidths=0.70,
    zorder=3,
)


ax.axvline(
    0,
    color=BLACK,
    linewidth=1,
    linestyle="--",
)


ax.axhline(
    0,
    color=BLACK,
    linewidth=1,
    linestyle="--",
)


cb = fig.colorbar(
    sc,
    ax=ax,
    pad=0.02,
)


cb.set_label(
    "Semantic threshold",
    fontsize=16,
)


cb.ax.tick_params(
    labelsize=13,
)


def nearest_point(
    f1_threshold,
    semantic_threshold,
):
    dist = (
        (
            wide[
                "content_f1_threshold"
            ]
            -
            f1_threshold
        ) ** 2
        +
        (
            wide[
                "semantic_threshold"
            ]
            -
            semantic_threshold
        ) ** 2
    )

    return wide.loc[
        dist.idxmin()
    ]


old = nearest_point(
    0.30,
    0.50,
)


cal_target = nearest_point(
    float(
        thresholds[
            "target"
        ][
            "content_f1"
        ]
    ),
    float(
        thresholds[
            "target"
        ][
            "semantic"
        ]
    ),
)


special = [
    (
        old,
        "Old 0.30 / 0.50",
        "X",
        "white",
    ),
    (
        cal_target,
        "Grouped-calibrated",
        "*",
        "#ffd54f",
    ),
]


for row, label, marker, color in special:
    xx = (
        100
        * row[
            "drr_target"
        ]
    )

    yy = (
        100
        * row[
            "irr_target"
        ]
    )

    ax.scatter(
        [xx],
        [yy],
        s=430,
        marker=marker,
        c=color,
        edgecolors=RED,
        linewidths=2.0,
        zorder=9,
    )

    ax.text(
        xx,
        yy + 1.45,
        (
            f"{label}\n"
            f"DRR {xx:+.1f}, "
            f"IRR {yy:+.1f} pp"
        ),
        ha="center",
        va="bottom",
        fontsize=13.4,
        fontweight="bold",
        bbox=callout_box(),
        zorder=10,
    )


kappa_min = float(
    agreement[
        "cohen_kappa"
    ].min()
)


kappa_max = float(
    agreement[
        "cohen_kappa"
    ].max()
)


target_test = (
    thresholds[
        "target"
    ][
        "calibrated_test_metrics"
    ]
)


system_test = (
    thresholds[
        "system"
    ][
        "calibrated_test_metrics"
    ]
)


ax.text(
    0.018,
    0.982,
    (
        f"Judge κ: "
        f"{kappa_min:.3f}–{kappa_max:.3f}\n"
        f"Target held-out F1: "
        f"{target_test['f1']:.3f}\n"
        f"System held-out F1: "
        f"{system_test['f1']:.3f}\n"
        f"Point size = content-F1 threshold"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14.3,
    fontweight="bold",
    bbox=callout_box(),
    zorder=10,
)


ax.set_xlabel(
    "Communication-defense Δ Target DRR (pp)"
)


ax.set_ylabel(
    "Communication-defense Δ Target IRR (pp)"
)


ax.set_title(
    "Threshold robustness atlas",
    fontweight="bold",
)


beautify(
    ax,
    "both",
)


fig.tight_layout()

save(
    fig,
    "07_threshold_robustness_atlas",
)


# ============================================================
# Index
# ============================================================

index = """
AgentUnlearn — final seven-figure package
=========================================

01_causal_trajectory
Overall causal path:
baseline -> local suppression -> communication-aware defense.

02_communication_defense_forest
Confirmatory communication-defense effects with
bootstrap 95% CIs and Holm-adjusted p-values.

03_topology_signature
Topology-specific fingerprint of communication defense.
Includes the independent-topology negative control.

04_benchmark_signature
Benchmark-specific effect matrix across all six
target/system outcomes.

05_model_and_parameter_landscape
Model-family consistency plus parameter-level
forgetting/retention trade-off.

06_robustness_and_wmdp
Seed/temperature robustness and WMDP appendix stress test.

07_threshold_robustness_atlas
Scorer threshold sensitivity, old vs grouped-calibrated
operating point, judge agreement, and held-out F1.

Visual language
---------------
- oversized typography;
- thin black structural outlines;
- translucent viridis gradients;
- divergent effect matrices where sign matters;
- white numeric callouts with dark-red borders;
- no default matplotlib bar-chart styling;
- vector PDF + 500-dpi PNG.
""".strip()


(
    OUT
    / "FIGURE_INDEX.txt"
).write_text(
    index
)


print()
print("=" * 92)
print("FINAL FIGURE PACKAGE COMPLETE")
print("=" * 92)
print("OUTPUT:", OUT)

for p in sorted(
    OUT.iterdir()
):
    print(p.name)
