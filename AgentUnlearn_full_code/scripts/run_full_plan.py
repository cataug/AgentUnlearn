#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/home/tahiti/AgentUnlearn")
CODE = ROOT / "AgentUnlearn_full_code"

DEFAULT_PYTHON = Path(
    "/home/tahiti/ARTeccv/.venv/bin/python"
)

# Multiple workers are used only WITHIN one model.
# Different models are processed sequentially to avoid stupid VRAM contention.
DEFAULT_WORKERS = {
    "qwen3_4b": 3,
    "qwen3_8b": 2,
    "mistral_7b": 2,
    "qwen25coder_1p5b": 4,
    "qwen25coder_7b": 2,
    "qwen3_14b": 1,
    "qwen25coder_14b": 1,
}

MODEL_ORDER = [
    "qwen3_4b",
    "qwen3_8b",
    "mistral_7b",
    "qwen25coder_1p5b",
    "qwen25coder_7b",
    "qwen3_14b",
    "qwen25coder_14b",
]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )

    tmp.replace(path)


def stable_shard(run_id: str, n: int) -> int:
    h = hashlib.sha1(
        str(run_id).encode("utf-8")
    ).hexdigest()

    return int(h[:12], 16) % n


def parse_worker_overrides(values):
    result = dict(DEFAULT_WORKERS)

    for value in values or []:
        if "=" not in value:
            raise ValueError(
                "--workers must look like qwen3_4b=3"
            )

        model, count = value.split("=", 1)
        model = model.strip()
        count = int(count)

        if count < 1:
            raise ValueError(
                f"Worker count must be >=1: {value}"
            )

        result[model] = count

    return result


def run_ids_complete(df, raw_dir: Path):
    return {
        str(rid)
        for rid in df["run_id"]
        if (raw_dir / f"{rid}.json").exists()
    }


def status_for_df(
    df,
    raw_dir: Path,
    error_dir: Path,
):
    ids = [
        str(x)
        for x in df["run_id"]
    ]

    complete = [
        rid
        for rid in ids
        if (raw_dir / f"{rid}.json").exists()
    ]

    errors = [
        rid
        for rid in ids
        if (
            (error_dir / f"{rid}.json").exists()
            and not (raw_dir / f"{rid}.json").exists()
        )
    ]

    return {
        "expected": len(ids),
        "complete": len(complete),
        "errors": len(errors),
        "remaining": len(ids) - len(complete),
        "error_ids": errors,
    }


def training_status(
    manifest: Path,
    checkpoints_dir: Path,
):
    df = pd.read_csv(manifest)

    complete = []
    resumable = []
    errors = []

    for tid in df["training_id"].astype(str):
        d = checkpoints_dir / tid

        if (d / "metadata.json").exists():
            complete.append(tid)

        elif (d / "resume.pt").exists():
            resumable.append(tid)

        if (
            (d / "error.json").exists()
            and tid not in complete
        ):
            errors.append(tid)

    return {
        "expected": len(df),
        "complete": len(complete),
        "resumable": len(resumable),
        "errors": len(errors),
        "remaining": len(df) - len(complete),
        "error_ids": errors,
    }


def pump_output(
    process,
    prefix: str,
    log_path: Path,
):
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log:

        for line in iter(
            process.stdout.readline,
            "",
        ):
            if not line:
                break

            log.write(line)
            log.flush()

            print(
                f"[{prefix}] {line}",
                end="",
                flush=True,
            )


def spawn_worker(
    python: Path,
    config: Path,
    manifest: Path,
    prefix: str,
    log_path: Path,
    env: dict,
):
    cmd = [
        str(python),
        "-u",
        str(
            CODE
            / "scripts"
            / "run_manifest.py"
        ),
        "--config",
        str(config),
        "--manifest",
        str(manifest),
    ]

    p = subprocess.Popen(
        cmd,
        cwd=str(CODE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    t = threading.Thread(
        target=pump_output,
        args=(
            p,
            prefix,
            log_path,
        ),
        daemon=True,
    )

    t.start()

    return p, t


def write_model_shards(
    df,
    phase: str,
    model_id: str,
    n_workers: int,
    shard_root: Path,
):
    mdf = (
        df[
            df["model_id"] == model_id
        ]
        .copy()
        .sort_values("run_id")
    )

    n_workers = min(
        n_workers,
        max(1, len(mdf)),
    )

    mdf["_worker"] = [
        stable_shard(
            rid,
            n_workers,
        )
        for rid in mdf["run_id"].astype(str)
    ]

    paths = []

    model_root = (
        shard_root
        / phase
        / model_id
    )

    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for worker in range(n_workers):
        shard = (
            mdf[
                mdf["_worker"] == worker
            ]
            .drop(
                columns=["_worker"]
            )
        )

        if len(shard) == 0:
            continue

        path = (
            model_root
            / f"worker_{worker:02d}.csv"
        )

        shard.to_csv(
            path,
            index=False,
        )

        paths.append(
            (
                worker,
                path,
                len(shard),
            )
        )

    return paths


def run_one_model_parallel(
    df,
    phase: str,
    model_id: str,
    workers: int,
    python: Path,
    config: Path,
    outputs: Path,
    raw_dir: Path,
    error_dir: Path,
    env: dict,
):
    mdf = df[
        df["model_id"] == model_id
    ].copy()

    before = status_for_df(
        mdf,
        raw_dir,
        error_dir,
    )

    if before["remaining"] == 0:
        print(
            f"\n[{phase}] {model_id}: "
            f"already complete "
            f"{before['complete']}/{before['expected']}",
            flush=True,
        )
        return before

    workers = min(
        int(workers),
        len(mdf),
    )

    print(
        "\n"
        + "=" * 96
    )

    print(
        f"{phase}: {model_id}"
    )

    print(
        f"conditions : {len(mdf)}"
    )

    print(
        f"complete   : {before['complete']}"
    )

    print(
        f"remaining  : {before['remaining']}"
    )

    print(
        f"workers    : {workers}"
    )

    print(
        "=" * 96,
        flush=True,
    )

    shard_root = (
        outputs
        / "parallel_shards"
    )

    shards = write_model_shards(
        mdf,
        phase,
        model_id,
        workers,
        shard_root,
    )

    running = []

    log_root = (
        outputs
        / "worker_logs"
        / phase
    )

    for worker, manifest, count in shards:

        prefix = (
            f"{model_id}:w{worker}"
        )

        log_path = (
            log_root
            / f"{model_id}_w{worker:02d}.log"
        )

        print(
            f"START {prefix}: "
            f"{count} assigned conditions",
            flush=True,
        )

        process, thread = spawn_worker(
            python=python,
            config=config,
            manifest=manifest,
            prefix=prefix,
            log_path=log_path,
            env=env,
        )

        running.append(
            (
                worker,
                process,
                thread,
            )
        )

    return_codes = []

    for worker, process, thread in running:

        code = process.wait()
        thread.join()

        return_codes.append(
            (
                worker,
                code,
            )
        )

        print(
            f"EXIT {model_id}:w{worker} "
            f"code={code}",
            flush=True,
        )

    after = status_for_df(
        mdf,
        raw_dir,
        error_dir,
    )

    print(
        f"\n{model_id} RESULT: "
        f"{after['complete']}/{after['expected']} complete, "
        f"{after['errors']} errors, "
        f"{after['remaining']} remaining",
        flush=True,
    )

    bad_codes = [
        x
        for x in return_codes
        if x[1] != 0
    ]

    if bad_codes:
        print(
            "Worker processes with non-zero "
            f"exit codes: {bad_codes}",
            flush=True,
        )

    return after


def model_sequence(df):
    existing = list(
        df["model_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    result = [
        m
        for m in MODEL_ORDER
        if m in existing
    ]

    result.extend(
        sorted(
            set(existing)
            - set(result)
        )
    )

    return result


def run_parallel_phase(
    df,
    phase: str,
    worker_map: dict,
    python: Path,
    config: Path,
    outputs: Path,
    raw_dir: Path,
    error_dir: Path,
    env: dict,
    retry_passes: int,
):
    total_passes = (
        1 + retry_passes
    )

    for pass_idx in range(
        total_passes
    ):

        overall = status_for_df(
            df,
            raw_dir,
            error_dir,
        )

        if overall["remaining"] == 0:
            return overall

        if pass_idx > 0:
            print(
                "\n"
                + "#" * 96
            )

            print(
                f"{phase}: RETRY PASS "
                f"{pass_idx}/{retry_passes}"
            )

            print(
                "#" * 96,
                flush=True,
            )

        for model_id in model_sequence(df):

            mdf = df[
                df["model_id"] == model_id
            ]

            st = status_for_df(
                mdf,
                raw_dir,
                error_dir,
            )

            if st["remaining"] == 0:
                continue

            n_workers = worker_map.get(
                model_id,
                1,
            )

            run_one_model_parallel(
                df=df,
                phase=phase,
                model_id=model_id,
                workers=n_workers,
                python=python,
                config=config,
                outputs=outputs,
                raw_dir=raw_dir,
                error_dir=error_dir,
                env=env,
            )

    return status_for_df(
        df,
        raw_dir,
        error_dir,
    )


def run_foreground(
    cmd,
    label,
    env,
):
    print(
        "\n"
        + "=" * 96
    )

    print(label)

    print(
        "=" * 96,
        flush=True,
    )

    result = subprocess.run(
        cmd,
        cwd=str(CODE),
        env=env,
    )

    return result.returncode


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default=str(
            CODE
            / "config"
            / "production.json"
        ),
    )

    ap.add_argument(
        "--python",
        default=str(
            DEFAULT_PYTHON
        ),
    )

    ap.add_argument(
        "--retry-passes",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--workers",
        action="append",
        default=[],
        help=(
            "Override worker count, e.g. "
            "--workers qwen3_4b=2"
        ),
    )

    ap.add_argument(
        "--no-score",
        action="store_true",
    )

    args = ap.parse_args()

    # IMPORTANT: do NOT call .resolve() here.
    # The venv interpreter is a symlink; resolving it dereferences
    # /home/tahiti/ARTeccv/.venv/bin/python -> /usr/bin/python3.10
    # and loses the virtual environment.
    python = Path(
        args.python
    )

    config = Path(
        args.config
    ).resolve()

    # Fail before spawning hundreds of workers if the requested
    # interpreter is not actually the intended virtual environment.
    check = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import sys,numpy,torch;"
                "print(sys.executable);"
                "print(sys.prefix);"
                "print(numpy.__file__);"
                "print(torch.cuda.is_available())"
            ),
        ],
        capture_output=True,
        text=True,
    )

    print("\nPYTHON ENVIRONMENT")
    print(check.stdout, flush=True)

    if check.returncode != 0:
        print(check.stderr, flush=True)
        raise RuntimeError(
            "Selected Python environment failed import sanity check"
        )

    if "/home/tahiti/ARTeccv/.venv" not in check.stdout:
        raise RuntimeError(
            "Workers are not using /home/tahiti/ARTeccv/.venv"
        )

    cfg = json.loads(
        config.read_text()
    )

    outputs = (
        ROOT
        / cfg["paths"]["outputs_dir"]
    ).resolve()

    checkpoints = (
        ROOT
        / cfg["paths"]["checkpoints_dir"]
    ).resolve()

    outputs.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dir = (
        outputs / "raw"
    )

    error_dir = (
        outputs / "errors"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_path = (
        outputs
        / "orchestrator_state.json"
    )

    worker_map = parse_worker_overrides(
        args.workers
    )

    print("\nWORKER MAP")

    for model, n in worker_map.items():
        print(
            f"  {model:22s} {n}"
        )

    free_gib = (
        shutil.disk_usage(outputs).free
        / 1024**3
    )

    print(
        f"\nFree disk: {free_gib:.1f} GiB"
    )

    # Prefer the immutable snapshot created by the first production run.
    snapshot = (
        outputs
        / "plan_snapshot"
    )

    snapshot_runs = (
        snapshot
        / "runs_all.csv"
    )

    if snapshot_runs.exists():
        runs_path = snapshot_runs
    else:
        runs_path = (
            ROOT
            / "manifests"
            / "runs_all.csv"
        )

    runs = pd.read_csv(
        runs_path
    )

    if runs["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate run_id in experiment plan"
        )

    training_manifest = (
        ROOT
        / "manifests"
        / "training_parameter_unlearning.csv"
    )

    pm = (
        runs["parameter_method"]
        .fillna("none")
        .astype(str)
    )

    nonparameter = (
        runs[
            pm == "none"
        ]
        .copy()
    )

    parameter = (
        runs[
            pm != "none"
        ]
        .copy()
    )

    print(
        "\nFULL PLAN"
    )

    print(
        f"  inference total : {len(runs)}"
    )

    print(
        f"  non-parameter   : {len(nonparameter)}"
    )

    print(
        f"  parameter eval  : {len(parameter)}"
    )

    print(
        f"  training jobs   : "
        f"{len(pd.read_csv(training_manifest))}"
    )

    state = {}

    if state_path.exists():
        try:
            state = json.loads(
                state_path.read_text()
            )
        except Exception:
            state = {}

    state.update({
        "updated_at": utcnow(),
        "worker_map": worker_map,
        "runs_manifest": str(runs_path),
    })

    atomic_json(
        state_path,
        state,
    )

    env = os.environ.copy()

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = (
        "expandable_segments:True"
    )

    # ==============================================================
    # 1. NON-PARAMETER INFERENCE
    # ==============================================================

    nonparam_status = (
        run_parallel_phase(
            df=nonparameter,
            phase="nonparameter",
            worker_map=worker_map,
            python=python,
            config=config,
            outputs=outputs,
            raw_dir=raw_dir,
            error_dir=error_dir,
            env=env,
            retry_passes=args.retry_passes,
        )
    )

    state[
        "nonparameter"
    ] = {
        **nonparam_status,
        "updated_at": utcnow(),
    }

    atomic_json(
        state_path,
        state,
    )

    print(
        "\nNON-PARAMETER FINAL:",
        nonparam_status,
    )

    # ==============================================================
    # 2. PARAMETER TRAINING
    #
    # Intentionally serial on one GPU. Gradient training has much
    # higher VRAM requirements than inference.
    # ==============================================================

    tstatus = training_status(
        training_manifest,
        checkpoints,
    )

    print(
        "\nPARAMETER TRAINING STATUS:",
        tstatus,
    )

    if tstatus["remaining"] > 0:

        for attempt in range(
            1 + args.retry_passes
        ):

            tstatus = training_status(
                training_manifest,
                checkpoints,
            )

            if tstatus["remaining"] == 0:
                break

            print(
                f"\nTraining pass "
                f"{attempt + 1}/"
                f"{1 + args.retry_passes}",
                flush=True,
            )

            run_foreground(
                [
                    str(python),
                    "-u",
                    str(
                        CODE
                        / "scripts"
                        / "train_parameter_unlearning.py"
                    ),
                    "--config",
                    str(config),
                ],
                "PARAMETER TRAINING",
                env,
            )

    tstatus = training_status(
        training_manifest,
        checkpoints,
    )

    state[
        "parameter_training"
    ] = {
        **tstatus,
        "updated_at": utcnow(),
    }

    atomic_json(
        state_path,
        state,
    )

    print(
        "\nPARAMETER TRAINING FINAL:",
        tstatus,
    )

    # ==============================================================
    # 3. PARAMETER EVALUATION
    #
    # Again parallel within a model. Every run_id is assigned to
    # exactly one worker shard.
    # ==============================================================

    param_status = (
        run_parallel_phase(
            df=parameter,
            phase="parameter_eval",
            worker_map=worker_map,
            python=python,
            config=config,
            outputs=outputs,
            raw_dir=raw_dir,
            error_dir=error_dir,
            env=env,
            retry_passes=args.retry_passes,
        )
    )

    state[
        "parameter_evaluation"
    ] = {
        **param_status,
        "updated_at": utcnow(),
    }

    atomic_json(
        state_path,
        state,
    )

    # ==============================================================
    # OVERALL
    # ==============================================================

    overall = status_for_df(
        runs,
        raw_dir,
        error_dir,
    )

    state[
        "overall"
    ] = {
        **overall,
        "updated_at": utcnow(),
    }

    atomic_json(
        state_path,
        state,
    )

    print(
        "\n"
        + "=" * 96
    )

    print(
        "FINAL INFERENCE STATUS"
    )

    print(
        "=" * 96
    )

    print(
        json.dumps(
            overall,
            indent=2,
        )
    )

    print(
        "\nTRAINING STATUS"
    )

    print(
        json.dumps(
            tstatus,
            indent=2,
        )
    )

    if (
        overall["remaining"] > 0
        or tstatus["remaining"] > 0
    ):
        print(
            "\nNOT EVERYTHING IS COMPLETE."
        )

        print(
            "Run this exact same command again. "
            "Completed run_ids are skipped and "
            "training resumes from checkpoints."
        )

        raise SystemExit(2)

    if args.no_score:
        print(
            "\nGeneration complete. "
            "Scoring disabled by --no-score."
        )
        return

    # ==============================================================
    # 4. SCORING
    # ==============================================================

    run_foreground(
        [
            str(python),
            "-u",
            str(
                CODE
                / "scripts"
                / "score_results.py"
            ),
            "--config",
            str(config),
        ],
        "BASE SCORING",
        env,
    )

    content_script = (
        CODE
        / "scripts"
        / "rescore_content.py"
    )

    metrics_dir = (
        outputs / "metrics"
    )

    if content_script.exists():

        run_foreground(
            [
                str(python),
                "-u",
                str(content_script),
                "--metrics-dir",
                str(metrics_dir),
                "--raw-dir",
                str(raw_dir),
                "--content-f1-threshold",
                "0.30",
                "--semantic-threshold",
                "0.50",
            ],
            "CONTENT-AWARE SCORING",
            env,
        )

    state[
        "scoring"
    ] = {
        "complete": True,
        "updated_at": utcnow(),
    }

    atomic_json(
        state_path,
        state,
    )

    print(
        "\n"
        + "=" * 96
    )

    print(
        "FULL PLAN COMPLETE"
    )

    print(
        "=" * 96
    )

    print(
        f"runs       : "
        f"{overall['complete']}/"
        f"{overall['expected']}"
    )

    print(
        f"training   : "
        f"{tstatus['complete']}/"
        f"{tstatus['expected']}"
    )

    print(
        f"raw        : {raw_dir}"
    )

    print(
        f"checkpoints: {checkpoints}"
    )

    print(
        f"metrics    : {metrics_dir}"
    )


if __name__ == "__main__":
    main()
