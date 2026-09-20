#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import json
from pathlib import Path

from agentunlearn.config import ExperimentConfig
from agentunlearn.io import read_csv


def main():
    cfg_path = Path("/home/tahiti/AgentUnlearn/config/defaults.json")
    cfg = ExperimentConfig.from_json(cfg_path)
    checks = []

    for label, path in [
        ("models_manifest", cfg.paths.resolve(cfg.paths.models_manifest)),
        ("runs_manifest", cfg.paths.resolve(cfg.paths.runs_manifest)),
        ("training_manifest", cfg.paths.resolve(cfg.paths.training_manifest)),
        ("data_dir", cfg.paths.resolve(cfg.paths.data_dir)),
    ]:
        ok = path.exists()
        checks.append((label, ok, str(path)))

    if checks[0][1]:
        for r in read_csv(checks[0][2]):
            p = Path(r["model_path"])
            checks.append((f"model:{r['model_id']}", p.exists(), str(p)))

    if checks[1][1]:
        rows = read_csv(checks[1][2])
        ids = [r["run_id"] for r in rows]
        checks.append(("unique_run_ids", len(ids) == len(set(ids)), f"rows={len(ids)} unique={len(set(ids))}"))

    failed = False
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4s} {name:30s} {detail}")
        failed |= not ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
