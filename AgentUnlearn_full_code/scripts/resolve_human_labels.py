#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--a", default="human_label_a")
    ap.add_argument("--b", default="human_label_b")
    ap.add_argument("--adjudicated", default="human_label_adjudicated")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    labels = []
    for _, r in df.iterrows():
        a, b = r.get(args.a), r.get(args.b)
        adj = r.get(args.adjudicated) if args.adjudicated in df.columns else None
        if pd.notna(adj):
            labels.append(int(adj)); continue
        if pd.notna(a) and pd.notna(b) and int(a) == int(b):
            labels.append(int(a)); continue
        labels.append(float("nan"))
    df["human_label"] = labels
    df.to_csv(args.output, index=False)
    unresolved = int(df["human_label"].isna().sum())
    print(f"wrote {args.output}; unresolved disagreements={unresolved}")


if __name__ == "__main__":
    main()
