#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd


def kappa(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    po = float(np.mean(a == b))
    pe = 0.0
    for lab in labels:
        pe += float(np.mean(a == lab) * np.mean(b == lab))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--annotator-a", default="human_label_a")
    ap.add_argument("--annotator-b", default="human_label_b")
    args = ap.parse_args()
    df = pd.read_csv(args.csv).dropna(subset=[args.annotator_a, args.annotator_b])
    a = df[args.annotator_a].astype(int).to_numpy()
    b = df[args.annotator_b].astype(int).to_numpy()
    print(json.dumps({"n": len(a), "agreement": float(np.mean(a == b)), "cohen_kappa": kappa(a, b)}, indent=2))


if __name__ == "__main__":
    main()
