#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse

from agentunlearn.config import ExperimentConfig
from agentunlearn.datasets import get_adapter
from agentunlearn.io import read_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/tahiti/AgentUnlearn/config/defaults.json")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    units = read_csv(cfg.paths.resolve("manifests/benchmark_units_manifest.csv"))
    data_dir = cfg.paths.resolve(cfg.paths.data_dir)
    for u in units:
        try:
            b = get_adapter(u["benchmark"], data_dir).load_bundle(u["unit_id"], u["target_id"], u["target_name"])
            print(
                f"{u['unit_id']:18s} direct={len(b.direct_probes):5d} indirect={len(b.indirect_probes):5d} "
                f"retain={len(b.retain_probes):5d} target_ev={len(b.target_evidence):5d} retain_ev={len(b.retain_evidence):5d}"
            )
        except Exception as e:
            print(f"{u['unit_id']:18s} FAIL {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
