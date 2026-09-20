from pathlib import Path
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--semantic-threshold", type=float, default=0.50)
    ap.add_argument("--token-f1-threshold", type=float, default=0.40)
    args = ap.parse_args()

    root = Path(args.metrics_dir)

    src = root / "probe_scores.csv"
    df = pd.read_csv(src)

    sem_tau = args.semantic_threshold
    f1_tau = args.token_f1_threshold

    # A reconstruction is accepted if:
    # 1) exact match succeeds, OR
    # 2) BOTH lexical and semantic evidence are sufficiently strong.
    #
    # This deliberately prevents generic same-entity answers from being
    # counted merely because their embedding is close to the reference.
    df["target_positive_conservative"] = (
        (df["target_exact"] >= 1)
        |
        (
            (df["target_token_f1"] >= f1_tau)
            &
            (df["target_semantic"] >= sem_tau)
        )
    ).astype(int)

    df["system_positive_conservative"] = (
        (df["system_exact"] >= 1)
        |
        (
            (df["system_token_f1"] >= f1_tau)
            &
            (df["system_semantic"] >= sem_tau)
        )
    ).astype(int)

    out_probe = root / "probe_scores_conservative.csv"
    df.to_csv(out_probe, index=False)

    keys = [
        "run_id",
        "block",
        "model_id",
        "benchmark",
        "unit_id",
        "topology",
        "intervention",
        "parameter_method",
    ]

    rows = []

    for values, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, values))

        def rate(kind, column):
            x = g[g["probe_kind"] == kind]
            if len(x) == 0:
                return float("nan")
            return float(x[column].mean())

        row.update({
            "drr_target":
                rate("direct", "target_positive_conservative"),
            "irr_target":
                rate("indirect", "target_positive_conservative"),
            "retention_target":
                rate("retain", "target_positive_conservative"),

            "drr_system":
                rate("direct", "system_positive_conservative"),
            "irr_system":
                rate("indirect", "system_positive_conservative"),
            "retention_system":
                rate("retain", "system_positive_conservative"),

            "n_scored_probes": len(g),
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    out_run = root / "run_metrics_conservative.csv"
    out.to_csv(out_run, index=False)

    print("WROTE:", out_probe)
    print("WROTE:", out_run)
    print()
    print(
        out[
            [
                "run_id",
                "topology",
                "intervention",
                "drr_target",
                "irr_target",
                "retention_target",
                "drr_system",
                "irr_system",
                "retention_system",
            ]
        ]
        .sort_values(["topology", "intervention"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
