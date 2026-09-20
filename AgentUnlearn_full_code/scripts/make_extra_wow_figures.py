#!/usr/bin/env python3

from pathlib import Path
import json
import math
import zlib

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm


ROOT = Path("/home/tahiti/AgentUnlearn")

AUTO = ROOT / "STATS/10_calibration/auto_judge"
STATS = ROOT / "STATS"
OUT = ROOT / "FIGURES"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data
# ============================================================

runs = pd.read_csv(
    AUTO / "run_metrics_calibrated.csv"
)

primary = pd.read_csv(
    AUTO / "PRIMARY_18_CALIBRATED.csv"
)

judged = pd.read_csv(
    AUTO / "judge_results_consensus.csv"
)

thresholds = json.loads(
    (
        AUTO
        / "calibrated_thresholds.json"
    ).read_text()
)


# ============================================================
# Style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": 21,

    "axes.titlesize": 27,
    "axes.labelsize": 23,

    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    "legend.fontsize": 16,

    "axes.linewidth": 1.15,
    "lines.linewidth": 2.4,

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
    lw=1.2,
):
    return dict(
        boxstyle=f"round,pad={pad}",
        fc="white",
        ec=RED,
        lw=lw,
        alpha=0.97,
    )


def save(fig, stem):
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
    )

    fig.savefig(
        OUT / f"{stem}.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    plt.close(fig)

    print("WROTE:", stem)


def clean(ax, axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis=axis,
        linewidth=0.7,
        color=BLACK,
        alpha=0.14,
    )

    ax.set_axisbelow(True)


def stable_seed(*x):
    s = "|".join(map(str, x))

    return (
        zlib.crc32(
            s.encode()
        )
        & 0xffffffff
    )


def boot(x, seed=42, n=5000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if not len(x):
        return np.nan, np.nan

    if len(x) == 1:
        return x[0], x[0]

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(n, len(x)),
    )

    b = x[idx].mean(axis=1)

    return (
        np.quantile(b, 0.025),
        np.quantile(b, 0.975),
    )


METRICS = [
    "drr_target",
    "irr_target",
    "retention_target",
    "drr_system",
    "irr_system",
    "retention_system",
]


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


main = runs[
    runs["block"]
    .astype(str)
    .eq("main")
].copy()


# ============================================================
# Helpers for communication-defense paired effects
# ============================================================

def communication_effects(
    group_cols,
):
    x = main[
        main["intervention"].isin([
            "target_context_memory",
            "full_inference",
        ])
    ].copy()

    rows = []

    group_cols = list(group_cols)

    for group_value, g in x.groupby(
        group_cols,
        dropna=False,
    ):
        if not isinstance(
            group_value,
            tuple,
        ):
            group_value = (
                group_value,
            )

        keys = [
            c
            for c in PAIR_KEYS
            if c in g.columns
            and c not in group_cols
        ]

        g = g.copy()

        for c in keys:
            g[c] = (
                g[c]
                .astype(object)
                .where(
                    g[c].notna(),
                    "__NA__",
                )
            )

        for metric in METRICS:
            t = g[
                keys
                + [
                    "intervention",
                    metric,
                ]
            ].dropna(
                subset=[metric]
            )

            p = t.pivot_table(
                index=keys,
                columns="intervention",
                values=metric,
                aggfunc="mean",
            )

            needed = [
                "target_context_memory",
                "full_inference",
            ]

            if not all(
                c in p.columns
                for c in needed
            ):
                continue

            p = p[
                needed
            ].dropna()

            if p.empty:
                continue

            d = (
                p["full_inference"]
                -
                p["target_context_memory"]
            ).to_numpy(float)

            lo, hi = boot(
                d,
                seed=stable_seed(
                    *group_value,
                    metric,
                ),
            )

            row = {
                c: v
                for c, v
                in zip(
                    group_cols,
                    group_value,
                )
            }

            row.update({
                "metric": metric,
                "delta": d.mean(),
                "lo": lo,
                "hi": hi,
                "n": len(d),
            })

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 08 — KNOWLEDGE ROUTING BRAID
# ============================================================

print("[08] knowledge routing braid")

topo_order = [
    "independent",
    "sequential",
    "debate",
    "scratchpad",
    "hierarchical",
]


braid_data = (
    main[
        main["intervention"].isin([
            "target_context_memory",
            "full_inference",
        ])
    ]
    .groupby(
        [
            "intervention",
            "topology",
        ]
    )[
        [
            "drr_target",
            "drr_system",
            "irr_target",
            "irr_system",
        ]
    ]
    .mean()
    * 100
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.8, 7.0),
    sharey=True,
)


interventions = [
    (
        "target_context_memory",
        "Local suppression",
    ),
    (
        "full_inference",
        "Local + communication",
    ),
]


for ax, (
    intervention,
    title,
) in zip(
    axes,
    interventions,
):

    colors = VIR(
        np.linspace(
            0.12,
            0.90,
            len(topo_order),
        )
    )

    for idx, (
        topo,
        color,
    ) in enumerate(
        zip(
            topo_order,
            colors,
        )
    ):
        key = (
            intervention,
            topo,
        )

        if key not in braid_data.index:
            continue

        r = braid_data.loc[key]

        # Direct reconstruction
        ax.plot(
            [0, 1],
            [
                r["drr_target"],
                r["drr_system"],
            ],
            marker="o",
            markersize=9,
            color=color,
            markeredgecolor=BLACK,
            markeredgewidth=1.0,
            alpha=0.82,
            linewidth=2.8,
        )

        # Indirect reconstruction
        ax.plot(
            [0, 1],
            [
                r["irr_target"],
                r["irr_system"],
            ],
            marker="s",
            markersize=7,
            color=color,
            markeredgecolor=BLACK,
            markeredgewidth=0.8,
            alpha=0.48,
            linewidth=1.7,
            linestyle="--",
        )

        gap = (
            r["drr_system"]
            -
            r["drr_target"]
        )

        ax.text(
            1.035,
            r["drr_system"],
            (
                f"{topo.replace('_',' ').title()}\n"
                f"gap {gap:+.1f} pp"
            ),
            fontsize=12.8,
            va="center",
            bbox=bbox(
                pad=0.13,
                lw=0.85,
            ),
        )

    ax.set_xlim(
        -0.15,
        1.74,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xticks(
        [0, 1],
        [
            "Target\nagent",
            "Whole\nsystem",
        ],
    )

    ax.set_title(
        title,
        fontweight="bold",
    )

    clean(
        ax,
        "y",
    )


axes[0].set_ylabel(
    "Accessible knowledge rate (%)"
)


fig.suptitle(
    "Knowledge-routing braid: suppression at the target does not imply system-level erasure",
    fontsize=26,
    fontweight="bold",
    y=1.02,
)


fig.text(
    0.5,
    -0.012,
    "Solid circles = direct recall; dashed squares = indirect reconstruction",
    ha="center",
    fontsize=16,
    color=GREY,
)


fig.tight_layout()

save(
    fig,
    "08_knowledge_routing_braid",
)


# ============================================================
# 09 — RECONSTRUCTION AMPLIFICATION MATRIX
# ============================================================

print("[09] reconstruction amplification matrix")

all_interventions = list(
    main[
        "intervention"
    ].drop_duplicates()
)


amp_rows = []


for (
    topology,
    intervention,
), g in main.groupby(
    [
        "topology",
        "intervention",
    ],
    dropna=False,
):

    amp_rows.append({
        "topology": topology,
        "intervention": intervention,

        "DRR amplification":
            100
            * (
                g["drr_system"].mean()
                -
                g["drr_target"].mean()
            ),

        "IRR amplification":
            100
            * (
                g["irr_system"].mean()
                -
                g["irr_target"].mean()
            ),
    })


amp = pd.DataFrame(
    amp_rows
)


columns = []

for intervention in all_interventions:
    columns += [
        (
            intervention,
            "DRR",
        ),
        (
            intervention,
            "IRR",
        ),
    ]


matrix = np.full(
    (
        len(topo_order),
        len(columns),
    ),
    np.nan,
)


for i, topology in enumerate(
    topo_order
):
    for j, (
        intervention,
        kind,
    ) in enumerate(
        columns
    ):
        z = amp[
            (
                amp["topology"]
                == topology
            )
            &
            (
                amp["intervention"]
                == intervention
            )
        ]

        if z.empty:
            continue

        col = (
            "DRR amplification"
            if kind == "DRR"
            else "IRR amplification"
        )

        matrix[
            i,
            j,
        ] = z.iloc[0][col]


finite = matrix[
    np.isfinite(matrix)
]

lim = (
    max(
        5,
        np.max(
            np.abs(finite)
        ),
    )
    if len(finite)
    else 5
)


fig, ax = plt.subplots(
    figsize=(
        max(
            14,
            len(columns) * 0.95,
        ),
        7.2,
    )
)


norm = TwoSlopeNorm(
    vmin=-lim,
    vcenter=0,
    vmax=lim,
)


im = ax.imshow(
    matrix,
    aspect="auto",
    cmap="PuOr_r",
    norm=norm,
    alpha=0.88,
)


ax.set_yticks(
    np.arange(
        len(topo_order)
    ),
    [
        x.title()
        for x in topo_order
    ],
)


xt = []

for intervention, kind in columns:
    short = (
        intervention
        .replace(
            "target_context_",
            "ctx+",
        )
        .replace(
            "target_memory",
            "mem",
        )
        .replace(
            "communication",
            "comm",
        )
        .replace(
            "full_inference",
            "full",
        )
        .replace(
            "global_context",
            "global",
        )
    )

    xt.append(
        f"{short}\n{kind}"
    )


ax.set_xticks(
    np.arange(
        len(columns)
    ),
    xt,
    rotation=42,
    ha="right",
)


ax.set_xticks(
    np.arange(
        -0.5,
        len(columns),
        1,
    ),
    minor=True,
)


ax.set_yticks(
    np.arange(
        -0.5,
        len(topo_order),
        1,
    ),
    minor=True,
)


ax.grid(
    which="minor",
    color=BLACK,
    linewidth=0.65,
    alpha=0.65,
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
        v = matrix[i, j]

        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:+.0f}",
                ha="center",
                va="center",
                fontsize=14.5,
                fontweight="bold",
                bbox=bbox(
                    pad=0.08,
                    lw=0.75,
                ),
            )


cb = fig.colorbar(
    im,
    ax=ax,
    pad=0.01,
)


cb.set_label(
    "System − target reconstruction (pp)",
    fontsize=18,
)


ax.set_title(
    "Reconstruction amplification across topology × intervention",
    fontweight="bold",
)


fig.tight_layout()

save(
    fig,
    "09_reconstruction_amplification_matrix",
)


# ============================================================
# 10 — INTERVENTION PHASE PORTRAIT
# ============================================================

print("[10] intervention phase portrait")


phase = (
    main.groupby(
        "intervention"
    )[
        [
            "drr_target",
            "irr_target",
            "retention_target",
            "drr_system",
        ]
    ]
    .mean()
    * 100
)


fig, ax = plt.subplots(
    figsize=(10.6, 8.0)
)


colors = VIR(
    np.linspace(
        0.12,
        0.94,
        len(phase),
    )
)


none_xy = None


if "none" in phase.index:
    none_xy = (
        phase.loc[
            "none",
            "drr_target",
        ],
        phase.loc[
            "none",
            "retention_target",
        ],
    )


for (
    intervention,
    row,
), color in zip(
    phase.iterrows(),
    colors,
):

    x = row[
        "drr_target"
    ]

    y = row[
        "retention_target"
    ]

    irr = row[
        "irr_target"
    ]

    system = row[
        "drr_system"
    ]

    size = (
        180
        +
        7.0
        * irr
    )

    alpha = (
        0.92
        if intervention
        in [
            "full_inference",
            "target_context_memory",
            "none",
        ]
        else 0.70
    )

    ax.scatter(
        x,
        y,
        s=size,
        c=[color],
        edgecolors=BLACK,
        linewidths=1.3,
        alpha=alpha,
        zorder=5,
    )

    if none_xy is not None and intervention != "none":
        ax.add_patch(
            FancyArrowPatch(
                none_xy,
                (
                    x,
                    y,
                ),
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                alpha=0.20,
                linewidth=1.3,
                connectionstyle="arc3,rad=0.10",
                zorder=1,
            )
        )

    label = (
        intervention
        .replace(
            "target_",
            "",
        )
        .replace(
            "_",
            " ",
        )
    )

    ax.text(
        x,
        y + 2.0,
        (
            f"{label}\n"
            f"IRR {irr:.0f} | "
            f"S-DRR {system:.0f}"
        ),
        ha="center",
        va="bottom",
        fontsize=12.8,
        fontweight="bold",
        bbox=bbox(
            pad=0.14,
            lw=0.85,
        ),
        zorder=9,
    )


ax.set_xlabel(
    "Target direct recall (%)  ← stronger forgetting"
)


ax.set_ylabel(
    "Target retention (%)  ↑ preserved utility"
)


ax.text(
    0.018,
    0.025,
    "Bubble size = target IRR",
    transform=ax.transAxes,
    fontsize=16,
    fontweight="bold",
    color=GREY,
)


ax.set_title(
    "Intervention phase portrait: forgetting × utility × reconstruction",
    fontweight="bold",
)


clean(
    ax,
    "both",
)


fig.tight_layout()

save(
    fig,
    "10_intervention_phase_portrait",
)


# ============================================================
# 11 — FULL FACTORIAL INTERVENTION FINGERPRINT
# ============================================================

print("[11] full intervention fingerprint")


means = (
    main.groupby(
        "intervention"
    )[METRICS]
    .mean()
)


if "none" in means.index:
    delta = (
        means
        -
        means.loc[
            "none"
        ]
    ) * 100
else:
    delta = (
        means
        -
        means.mean()
    ) * 100


# Order by target direct suppression.
order = list(
    delta.sort_values(
        "drr_target"
    ).index
)


mat = delta.loc[
    order,
    METRICS,
].to_numpy()


finite = mat[
    np.isfinite(mat)
]


lim = max(
    1.0,
    np.max(
        np.abs(finite)
    ),
)


fig, ax = plt.subplots(
    figsize=(11.5, 8.2)
)


norm = TwoSlopeNorm(
    vmin=-lim,
    vcenter=0,
    vmax=lim,
)


im = ax.imshow(
    mat,
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
        SHORT[x]
        for x in METRICS
    ],
)


ax.set_yticks(
    np.arange(
        len(order)
    ),
    [
        x.replace(
            "_",
            " ",
        )
        for x in order
    ],
)


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
        len(order),
        1,
    ),
    minor=True,
)


ax.grid(
    which="minor",
    color=BLACK,
    linewidth=0.78,
    alpha=0.72,
)


ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


for i in range(
    len(order)
):
    for j in range(
        len(METRICS)
    ):
        v = mat[i, j]

        ax.text(
            j,
            i,
            f"{v:+.1f}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            bbox=bbox(
                pad=0.10,
                lw=0.82,
            ),
        )


cb = fig.colorbar(
    im,
    ax=ax,
    pad=0.018,
)


cb.set_label(
    "Change relative to no intervention (pp)",
    fontsize=18,
)


ax.set_title(
    "Full-factorial intervention fingerprint",
    fontweight="bold",
)


fig.tight_layout()

save(
    fig,
    "11_full_factorial_intervention_fingerprint",
)


# ============================================================
# 12 — MODEL × BENCHMARK BUBBLE ATLAS
# ============================================================

print("[12] model × benchmark bubble atlas")


mb = communication_effects(
    [
        "model_id",
        "benchmark",
    ]
)


models = list(
    mb[
        "model_id"
    ].drop_duplicates()
)


benchmarks = [
    b
    for b in [
        "TOFU",
        "RWKU",
        "MUSE-Books",
        "MUSE-News",
        "WMDP",
    ]
    if b in set(
        mb["benchmark"]
    )
]


fig, ax = plt.subplots(
    figsize=(
        11.5,
        max(
            6,
            2.1
            +
            1.0
            * len(models),
        ),
    )
)


values = mb[
    mb["metric"]
    .eq("drr_target")
][
    "delta"
].to_numpy()


if len(values):
    lim = max(
        0.03,
        np.nanmax(
            np.abs(values)
        ),
    )
else:
    lim = 0.1


norm = TwoSlopeNorm(
    vmin=-lim,
    vcenter=0,
    vmax=lim,
)


cmap = mpl.colormaps[
    "RdBu_r"
]


for iy, model in enumerate(
    models
):
    for ix, benchmark in enumerate(
        benchmarks
    ):

        drr = mb[
            (
                mb["model_id"]
                == model
            )
            &
            (
                mb["benchmark"]
                == benchmark
            )
            &
            (
                mb["metric"]
                == "drr_target"
            )
        ]

        irr = mb[
            (
                mb["model_id"]
                == model
            )
            &
            (
                mb["benchmark"]
                == benchmark
            )
            &
            (
                mb["metric"]
                == "irr_target"
            )
        ]

        if drr.empty:
            continue

        d = float(
            drr.iloc[0][
                "delta"
            ]
        )

        ir = (
            float(
                irr.iloc[0][
                    "delta"
                ]
            )
            if not irr.empty
            else np.nan
        )

        size = (
            360
            +
            2300
            * abs(d)
        )

        circle = ax.scatter(
            ix,
            iy,
            s=size,
            c=[
                cmap(
                    norm(d)
                )
            ],
            edgecolors=BLACK,
            linewidths=1.25,
            alpha=0.82,
            zorder=3,
        )

        txt = (
            f"D {100*d:+.1f}"
        )

        if np.isfinite(ir):
            txt += (
                f"\nI {100*ir:+.1f}"
            )

        ax.text(
            ix,
            iy,
            txt,
            ha="center",
            va="center",
            fontsize=13.7,
            fontweight="bold",
            bbox=bbox(
                pad=0.10,
                lw=0.75,
            ),
            zorder=7,
        )


ax.set_xticks(
    np.arange(
        len(benchmarks)
    ),
    benchmarks,
)


ax.set_yticks(
    np.arange(
        len(models)
    ),
    [
        m.replace(
            "_",
            " ",
        )
        for m in models
    ],
)


ax.set_xlim(
    -0.55,
    len(benchmarks) - 0.45,
)


ax.set_ylim(
    len(models) - 0.45,
    -0.55,
)


ax.set_xticks(
    np.arange(
        -0.5,
        len(benchmarks),
        1,
    ),
    minor=True,
)


ax.set_yticks(
    np.arange(
        -0.5,
        len(models),
        1,
    ),
    minor=True,
)


ax.grid(
    which="minor",
    color=BLACK,
    linewidth=0.65,
    alpha=0.34,
)


ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


sm = mpl.cm.ScalarMappable(
    norm=norm,
    cmap=cmap,
)


cb = fig.colorbar(
    sm,
    ax=ax,
    pad=0.02,
)


cb.set_label(
    "Δ target DRR",
    fontsize=18,
)


ax.set_title(
    "Model × benchmark communication-defense atlas",
    fontweight="bold",
)


ax.set_xlabel(
    "Bubble size = |DRR effect|; labels show DRR / IRR change in pp",
    fontsize=16,
)


fig.tight_layout()

save(
    fig,
    "12_model_benchmark_bubble_atlas",
)


# ============================================================
# 13 — JUDGE DECISION LANDSCAPE
# ============================================================

print("[13] judge decision landscape")


fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.4, 6.8),
    sharex=True,
    sharey=True,
)


for ax, side in zip(
    axes,
    [
        "target",
        "system",
    ]
):

    d = judged[
        (
            judged["side"]
            == side
        )
        &
        (
            judged[
                "judge_consensus"
            ].isin(
                [0, 1]
            )
        )
    ].copy()

    f1col = (
        f"{side}_content_f1"
    )

    semcol = (
        f"{side}_semantic"
    )

    exactcol = (
        f"{side}_exact"
    )

    d = d[
        d[
            exactcol
        ]
        < 0.5
    ].copy()

    tf1 = float(
        thresholds[
            side
        ][
            "content_f1"
        ]
    )

    tsem = float(
        thresholds[
            side
        ][
            "semantic"
        ]
    )

    # Background decision region.
    xx = np.linspace(
        0,
        max(
            1,
            d[
                f1col
            ].max(),
        ),
        240,
    )

    yy = np.linspace(
        0,
        max(
            1,
            d[
                semcol
            ].max(),
        ),
        240,
    )

    X, Y = np.meshgrid(
        xx,
        yy,
    )

    region = (
        (X >= tf1)
        &
        (Y >= tsem)
    ).astype(float)

    ax.imshow(
        region,
        extent=(
            xx.min(),
            xx.max(),
            yy.min(),
            yy.max(),
        ),
        origin="lower",
        cmap="viridis",
        alpha=0.12,
        aspect="auto",
        zorder=0,
    )

    for label, marker, color in [
        (
            0,
            "o",
            VIR(0.12),
        ),
        (
            1,
            "D",
            VIR(0.83),
        ),
    ]:
        z = d[
            d[
                "judge_consensus"
            ]
            == label
        ]

        ax.scatter(
            z[f1col],
            z[semcol],
            s=72,
            marker=marker,
            c=[color],
            edgecolors=BLACK,
            linewidths=0.7,
            alpha=0.66,
            label=(
                "Judge negative"
                if label == 0
                else "Judge positive"
            ),
            zorder=3,
        )

    ax.axvline(
        tf1,
        color=RED,
        linewidth=1.7,
        linestyle="--",
    )

    ax.axhline(
        tsem,
        color=RED,
        linewidth=1.7,
        linestyle="--",
    )

    metrics = thresholds[
        side
    ][
        "calibrated_test_metrics"
    ]

    ax.text(
        0.025,
        0.975,
        (
            f"τF1 = {tf1:.2f}\n"
            f"τsem = {tsem:.2f}\n"
            f"held-out F1 = "
            f"{metrics['f1']:.3f}\n"
            f"precision = "
            f"{metrics['precision']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=15,
        fontweight="bold",
        bbox=bbox(),
        zorder=9,
    )

    ax.set_title(
        side.title(),
        fontweight="bold",
    )

    ax.set_xlabel(
        "Content F1"
    )

    clean(
        ax,
        "both",
    )


axes[0].set_ylabel(
    "Semantic similarity"
)


axes[0].legend(
    frameon=False,
    loc="lower right",
)


fig.suptitle(
    "LLM-judge decision landscape and calibrated scoring boundary",
    fontsize=26,
    fontweight="bold",
    y=1.02,
)


fig.tight_layout()

save(
    fig,
    "13_judge_decision_landscape",
)


# ============================================================
# 14 — PARAMETER METHOD FINGERPRINT
# ============================================================

print("[14] parameter method fingerprint")


parameter = runs[
    ~(
        runs[
            "parameter_method"
        ]
        .astype(str)
        .str.lower()
        .isin([
            "none",
            "nan",
            "",
        ])
    )
].copy()


if len(parameter):

    methods = list(
        parameter[
            "parameter_method"
        ].drop_duplicates()
    )

    models = list(
        parameter[
            "model_id"
        ].drop_duplicates()
    )

    drr = (
        parameter
        .groupby(
            [
                "parameter_method",
                "model_id",
            ]
        )[
            "drr_target"
        ]
        .mean()
        .unstack()
        .reindex(
            index=methods,
            columns=models,
        )
        * 100
    )

    ret = (
        parameter
        .groupby(
            [
                "parameter_method",
                "model_id",
            ]
        )[
            "retention_target"
        ]
        .mean()
        .unstack()
        .reindex(
            index=methods,
            columns=models,
        )
        * 100
    )


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.3, 6.3),
        sharey=True,
    )


    for (
        ax,
        matrix,
        title,
        cmap,
    ) in [
        (
            axes[0],
            drr,
            "Target DRR ↓",
            "magma",
        ),
        (
            axes[1],
            ret,
            "Target retention ↑",
            "viridis",
        ),
    ]:

        arr = matrix.to_numpy()

        im = ax.imshow(
            arr,
            aspect="auto",
            cmap=cmap,
            alpha=0.86,
            vmin=0,
            vmax=100,
        )

        ax.set_xticks(
            np.arange(
                len(models)
            ),
            [
                x.replace(
                    "_",
                    " ",
                )
                for x in models
            ],
            rotation=22,
            ha="right",
        )

        ax.set_yticks(
            np.arange(
                len(methods)
            ),
            [
                x.replace(
                    "_",
                    " ",
                )
                for x in methods
            ],
        )

        ax.set_xticks(
            np.arange(
                -0.5,
                len(models),
                1,
            ),
            minor=True,
        )

        ax.set_yticks(
            np.arange(
                -0.5,
                len(methods),
                1,
            ),
            minor=True,
        )

        ax.grid(
            which="minor",
            color=BLACK,
            linewidth=0.75,
            alpha=0.72,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        for i in range(
            arr.shape[0]
        ):
            for j in range(
                arr.shape[1]
            ):
                v = arr[i, j]

                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.1f}",
                        ha="center",
                        va="center",
                        fontsize=17,
                        fontweight="bold",
                        bbox=bbox(
                            pad=0.11,
                            lw=0.8,
                        ),
                    )

        ax.set_title(
            title,
            fontweight="bold",
        )

        cb = fig.colorbar(
            im,
            ax=ax,
            pad=0.02,
        )

        cb.ax.tick_params(
            labelsize=14,
        )


    fig.suptitle(
        "Parameter-level unlearning fingerprint",
        fontsize=27,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()

    save(
        fig,
        "14_parameter_method_fingerprint",
    )

else:
    print(
        "SKIP 14: no parameter_method rows"
    )


# ============================================================
# 15 — TRAINING DYNAMICS (automatic column discovery)
# ============================================================

print("[15] training dynamics")


training_path = (
    STATS
    / "06_training"
    / "training_loss_history.csv"
)


if training_path.exists():

    hist = pd.read_csv(
        training_path
    )

    numeric_cols = [
        c
        for c in hist.columns
        if pd.api.types.is_numeric_dtype(
            hist[c]
        )
    ]

    step_candidates = [
        c
        for c in hist.columns
        if c.lower()
        in [
            "step",
            "global_step",
            "train_step",
            "iteration",
            "epoch_step",
        ]
    ]

    loss_candidates = [
        c
        for c in numeric_cols
        if "loss" in c.lower()
    ]

    id_candidates = [
        c
        for c in hist.columns
        if c.lower()
        in [
            "train_id",
            "training_id",
            "job_id",
            "run_id",
            "checkpoint_id",
        ]
    ]

    if (
        step_candidates
        and loss_candidates
    ):

        step_col = step_candidates[0]
        loss_col = loss_candidates[0]

        id_col = (
            id_candidates[0]
            if id_candidates
            else None
        )

        if id_col is None:
            hist["_curve"] = "training"
            id_col = "_curve"

        groups = list(
            hist.groupby(
                id_col,
                dropna=False,
            )
        )

        fig, ax = plt.subplots(
            figsize=(11.0, 7.2)
        )

        colors = VIR(
            np.linspace(
                0.10,
                0.92,
                max(
                    1,
                    len(groups),
                ),
            )
        )

        common_x = np.linspace(
            0,
            1,
            120,
        )

        curves = []

        for (
            name,
            g,
        ), color in zip(
            groups,
            colors,
        ):

            g = (
                g[
                    [
                        step_col,
                        loss_col,
                    ]
                ]
                .dropna()
                .sort_values(
                    step_col
                )
            )

            if len(g) < 2:
                continue

            x = g[
                step_col
            ].to_numpy(float)

            y = g[
                loss_col
            ].to_numpy(float)

            if x.max() == x.min():
                continue

            xn = (
                x - x.min()
            ) / (
                x.max() - x.min()
            )

            ax.plot(
                xn,
                y,
                color=color,
                alpha=0.28,
                linewidth=1.6,
            )

            curves.append(
                np.interp(
                    common_x,
                    xn,
                    y,
                )
            )

        if curves:

            curves = np.vstack(
                curves
            )

            median = np.nanmedian(
                curves,
                axis=0,
            )

            lo = np.nanquantile(
                curves,
                0.25,
                axis=0,
            )

            hi = np.nanquantile(
                curves,
                0.75,
                axis=0,
            )

            ax.fill_between(
                common_x,
                lo,
                hi,
                color=VIR(0.60),
                alpha=0.17,
                edgecolor=BLACK,
                linewidth=0.7,
            )

            ax.plot(
                common_x,
                median,
                color=BLACK,
                linewidth=3.0,
                label="Median trajectory",
            )

            ax.text(
                common_x[-1],
                median[-1],
                f" final {median[-1]:.3f}",
                va="center",
                ha="left",
                fontsize=16,
                fontweight="bold",
                bbox=bbox(),
            )

        ax.set_xlabel(
            "Normalized training progress"
        )

        ax.set_ylabel(
            loss_col.replace(
                "_",
                " ",
            ).title()
        )

        ax.set_title(
            "Parameter-unlearning optimization trajectories",
            fontweight="bold",
        )

        clean(
            ax,
            "both",
        )

        ax.legend(
            frameon=False,
        )

        fig.tight_layout()

        save(
            fig,
            "15_parameter_training_dynamics",
        )

    else:
        print(
            "SKIP 15: could not identify step/loss columns"
        )
        print(
            "columns:",
            list(hist.columns),
        )

else:
    print(
        "SKIP 15: training_loss_history.csv absent"
    )


print()
print("=" * 92)
print("EXTRA WOW FIGURES COMPLETE")
print("=" * 92)

for p in sorted(
    OUT.glob("*.pdf")
):
    print(p.name)

