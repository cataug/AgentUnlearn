#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/home/tahiti/AgentUnlearn"
)

CODE = ROOT / "AgentUnlearn_full_code"

DEFAULT_PYTHON = Path(
    "/home/tahiti/ARTeccv/"
    ".venv/bin/python"
)


def now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def atomic_json(
    path: Path,
    obj,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def sha256(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def run_command(
    cmd: list[str],
    label: str,
    cwd: Path,
    env: dict,
) -> None:

    print(
        "\n"
        + "=" * 88
    )

    print(
        f"STAGE: {label}"
    )

    print(
        "=" * 88
    )

    print(
        " ".join(cmd),
        flush=True,
    )

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{label} failed "
            f"with exit code "
            f"{result.returncode}"
        )


def inference_status(
    manifest: Path,
    raw_dir: Path,
    error_dir: Path,
) -> dict:

    df = pd.read_csv(
        manifest
    )

    ids = [
        str(x)
        for x in df["run_id"]
    ]

    complete = [
        x
        for x in ids
        if (
            raw_dir
            / f"{x}.json"
        ).exists()
    ]

    errors = [
        x
        for x in ids
        if (
            error_dir
            / f"{x}.json"
        ).exists()
        and not (
            raw_dir
            / f"{x}.json"
        ).exists()
    ]

    return {
        "expected": len(ids),
        "complete": len(complete),
        "errors": len(errors),
        "remaining": (
            len(ids)
            - len(complete)
        ),
        "error_ids": errors,
    }


def training_status(
    manifest: Path,
    checkpoints_dir: Path,
) -> dict:

    df = pd.read_csv(
        manifest
    )

    ids = [
        str(x)
        for x in df[
            "training_id"
        ]
    ]

    complete = []

    resumable = []

    errors = []

    for tid in ids:

        d = (
            checkpoints_dir
            / tid
        )

        if (
            d / "metadata.json"
        ).exists():

            complete.append(tid)

        elif (
            d / "resume.pt"
        ).exists():

            resumable.append(tid)

        if (
            d / "error.json"
        ).exists() and tid not in complete:

            errors.append(tid)

    return {
        "expected": len(ids),
        "complete": len(complete),
        "resumable": len(resumable),
        "errors": len(errors),
        "remaining": (
            len(ids)
            - len(complete)
        ),
        "error_ids": errors,
    }


def update_state(
    path: Path,
    state: dict,
    stage: str,
    payload: dict,
) -> None:

    state[
        "updated_at"
    ] = now()

    state[
        "stages"
    ][stage] = payload

    atomic_json(
        path,
        state,
    )


def copy_snapshot(
    snapshot_dir: Path,
    files: list[Path],
) -> dict:

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hashes = {}

    for src in files:

        if not src.exists():
            continue

        dst = (
            snapshot_dir
            / src.name
        )

        shutil.copy2(
            src,
            dst,
        )

        hashes[
            src.name
        ] = sha256(dst)

    atomic_json(
        snapshot_dir
        / "sha256.json",
        hashes,
    )

    return hashes


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
        "--no-score",
        action="store_true",
    )

    ap.add_argument(
        "--retry-passes",
        type=int,
        default=1,
        help=(
            "Automatic retry passes "
            "after the first inference/"
            "training pass."
        ),
    )

    args = ap.parse_args()

    python = Path(
        args.python
    )

    config_path = Path(
        args.config
    ).resolve()

    if not python.exists():
        raise RuntimeError(
            f"Python not found: {python}"
        )

    if not config_path.exists():
        raise RuntimeError(
            "Production config missing: "
            f"{config_path}"
        )

    cfg = json.loads(
        config_path.read_text()
    )

    outputs_rel = cfg[
        "paths"
    ][
        "outputs_dir"
    ]

    outputs = (
        ROOT
        / outputs_rel
    ).resolve()

    checkpoints_rel = cfg[
        "paths"
    ][
        "checkpoints_dir"
    ]

    checkpoints = (
        ROOT
        / checkpoints_rel
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

    state = {
        "created_at": now(),
        "updated_at": now(),
        "config":
            str(config_path),
        "outputs":
            str(outputs),
        "stages": {},
    }

    if state_path.exists():

        try:
            old = json.loads(
                state_path.read_text()
            )

            state.update(old)

        except Exception:
            pass

    # --------------------------------------------------------------
    # Disk warning
    # --------------------------------------------------------------

    usage = shutil.disk_usage(
        outputs
    )

    free_gib = (
        usage.free
        / 1024**3
    )

    print(
        f"Free disk: "
        f"{free_gib:.1f} GiB"
    )

    if free_gib < 40:

        print(
            "WARNING: less than "
            "40 GiB free. Parameter "
            "checkpoints may consume "
            "substantial disk space.",
            flush=True,
        )

    # --------------------------------------------------------------
    # Exact experiment manifests
    # --------------------------------------------------------------

    runs_all = (
        ROOT
        / "manifests"
        / "runs_all.csv"
    )

    training_manifest = (
        ROOT
        / "manifests"
        / "training_parameter_unlearning.csv"
    )

    models_manifest = (
        ROOT
        / "manifests"
        / "models_manifest.csv"
    )

    units_manifest = (
        ROOT
        / "manifests"
        / "benchmark_units_manifest.csv"
    )

    if not runs_all.exists():
        raise RuntimeError(
            f"Missing {runs_all}"
        )

    if not training_manifest.exists():
        raise RuntimeError(
            f"Missing {training_manifest}"
        )

    runs = pd.read_csv(
        runs_all
    )

    if runs[
        "run_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate run_id in "
            "runs_all.csv"
        )

    pm = (
        runs[
            "parameter_method"
        ]
        .fillna("none")
        .astype(str)
    )

    nonparameter = runs[
        pm == "none"
    ].copy()

    parameter = runs[
        pm != "none"
    ].copy()

    plan_dir = (
        outputs / "plan_snapshot"
    )

    plan_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nonparameter_manifest = (
        plan_dir
        / "runs_nonparameter.csv"
    )

    parameter_manifest = (
        plan_dir
        / "runs_parameter.csv"
    )

    nonparameter.to_csv(
        nonparameter_manifest,
        index=False,
    )

    parameter.to_csv(
        parameter_manifest,
        index=False,
    )

    # Exact immutable-ish snapshot of what this run used.
    snapshot_files = [
        config_path,
        runs_all,
        training_manifest,
        models_manifest,
        units_manifest,
        ROOT
        / "manifests"
        / "interventions_manifest.csv",
        ROOT
        / "manifests"
        / "topologies_manifest.csv",
    ]

    hashes = copy_snapshot(
        plan_dir,
        snapshot_files,
    )

    state[
        "plan"
    ] = {
        "inference_total":
            len(runs),
        "nonparameter_inference":
            len(nonparameter),
        "parameter_inference":
            len(parameter),
        "training_jobs":
            len(
                pd.read_csv(
                    training_manifest
                )
            ),
        "snapshot_hashes":
            hashes,
    }

    atomic_json(
        state_path,
        state,
    )

    print(
        "\n"
        "FULL PLAN\n"
        f"  inference total    : "
        f"{len(runs)}\n"
        f"  non-parameter      : "
        f"{len(nonparameter)}\n"
        f"  parameter eval     : "
        f"{len(parameter)}\n"
        f"  parameter training : "
        f"{len(pd.read_csv(training_manifest))}"
    )

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    env = os.environ.copy()

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = (
        "expandable_segments:True"
    )

    # --------------------------------------------------------------
    # STAGE 1: all non-parameter inference
    #
    # run_manifest groups by model_id, therefore each model is loaded
    # only once for all compatible conditions in this manifest.
    # --------------------------------------------------------------

    status = inference_status(
        nonparameter_manifest,
        raw_dir,
        error_dir,
    )

    print(
        "\nNON-PARAMETER STATUS:",
        status,
    )

    if status[
        "complete"
    ] < status[
        "expected"
    ]:

        run_command(
            [
                str(python),
                str(
                    CODE
                    / "scripts"
                    / "run_manifest.py"
                ),
                "--config",
                str(config_path),
                "--manifest",
                str(
                    nonparameter_manifest
                ),
            ],
            "NON-PARAMETER INFERENCE",
            CODE,
            env,
        )

    for retry in range(
        args.retry_passes
    ):

        status = inference_status(
            nonparameter_manifest,
            raw_dir,
            error_dir,
        )

        if (
            status["complete"]
            >= status["expected"]
        ):
            break

        print(
            "\nAutomatic retry "
            f"{retry + 1}/"
            f"{args.retry_passes}"
        )

        run_command(
            [
                str(python),
                str(
                    CODE
                    / "scripts"
                    / "run_manifest.py"
                ),
                "--config",
                str(config_path),
                "--manifest",
                str(
                    nonparameter_manifest
                ),
            ],
            (
                "NON-PARAMETER "
                f"RETRY {retry + 1}"
            ),
            CODE,
            env,
        )

    status = inference_status(
        nonparameter_manifest,
        raw_dir,
        error_dir,
    )

    update_state(
        state_path,
        state,
        "nonparameter_inference",
        {
            **status,
            "finished_at": now(),
        },
    )

    print(
        "\nNON-PARAMETER FINAL:",
        status,
    )

    # --------------------------------------------------------------
    # STAGE 2: parameter-unlearning training
    # --------------------------------------------------------------

    tstatus = training_status(
        training_manifest,
        checkpoints,
    )

    print(
        "\nTRAINING STATUS:",
        tstatus,
    )

    if (
        tstatus["complete"]
        < tstatus["expected"]
    ):

        run_command(
            [
                str(python),
                str(
                    CODE
                    / "scripts"
                    / "train_parameter_unlearning.py"
                ),
                "--config",
                str(config_path),
            ],
            "PARAMETER TRAINING",
            CODE,
            env,
        )

    for retry in range(
        args.retry_passes
    ):

        tstatus = training_status(
            training_manifest,
            checkpoints,
        )

        if (
            tstatus["complete"]
            >= tstatus["expected"]
        ):
            break

        print(
            "\nAutomatic training retry "
            f"{retry + 1}/"
            f"{args.retry_passes}"
        )

        run_command(
            [
                str(python),
                str(
                    CODE
                    / "scripts"
                    / "train_parameter_unlearning.py"
                ),
                "--config",
                str(config_path),
            ],
            (
                "PARAMETER TRAINING "
                f"RETRY {retry + 1}"
            ),
            CODE,
            env,
        )

    tstatus = training_status(
        training_manifest,
        checkpoints,
    )

    update_state(
        state_path,
        state,
        "parameter_training",
        {
            **tstatus,
            "finished_at": now(),
        },
    )

    print(
        "\nPARAMETER TRAINING FINAL:",
        tstatus,
    )

    # --------------------------------------------------------------
    # STAGE 3: parameter evaluation.
    #
    # Missing final checkpoints are automatically skipped by
    # run_manifest, so partial training never contaminates evaluation.
    # --------------------------------------------------------------

    pstatus = inference_status(
        parameter_manifest,
        raw_dir,
        error_dir,
    )

    if (
        pstatus["complete"]
        < pstatus["expected"]
    ):

        run_command(
            [
                str(python),
                str(
                    CODE
                    / "scripts"
                    / "run_manifest.py"
                ),
                "--config",
                str(config_path),
                "--manifest",
                str(
                    parameter_manifest
                ),
            ],
            "PARAMETER EVALUATION",
            CODE,
            env,
        )

    for retry in range(
        args.retry_passes
    ):

        pstatus = inference_status(
            parameter_manifest,
            raw_dir,
            error_dir,
        )

        if (
            pstatus["complete"]
            >= pstatus["expected"]
        ):
            break

        # Retry only makes sense if training checkpoints now exist.
        run_command(
            [
                str(python),
                str(
                    CODE
                    / "scripts"
                    / "run_manifest.py"
                ),
                "--config",
                str(config_path),
                "--manifest",
                str(
                    parameter_manifest
                ),
            ],
            (
                "PARAMETER EVALUATION "
                f"RETRY {retry + 1}"
            ),
            CODE,
            env,
        )

    pstatus = inference_status(
        parameter_manifest,
        raw_dir,
        error_dir,
    )

    update_state(
        state_path,
        state,
        "parameter_evaluation",
        {
            **pstatus,
            "finished_at": now(),
        },
    )

    # --------------------------------------------------------------
    # Overall inference status
    # --------------------------------------------------------------

    all_status = inference_status(
        runs_all,
        raw_dir,
        error_dir,
    )

    update_state(
        state_path,
        state,
        "all_inference",
        {
            **all_status,
            "finished_at": now(),
        },
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "INFERENCE FINAL STATUS"
    )

    print(
        "=" * 88
    )

    print(
        json.dumps(
            all_status,
            indent=2,
        )
    )

    # --------------------------------------------------------------
    # Only score a complete experiment.
    # If anything failed, rerunning this exact same command retries only
    # missing conditions / unfinished training jobs.
    # --------------------------------------------------------------

    if (
        all_status["complete"]
        != all_status["expected"]
        or tstatus["complete"]
        != tstatus["expected"]
    ):

        print(
            "\nFULL PLAN IS NOT YET COMPLETE."
        )

        print(
            "Rerun this same command. "
            "Completed jobs will be skipped "
            "and unfinished jobs resumed."
        )

        raise SystemExit(2)

    if args.no_score:

        print(
            "\nAll generation/training "
            "complete. Scoring skipped "
            "because --no-score was set."
        )

        return

    # --------------------------------------------------------------
    # STAGE 4: base scoring
    # --------------------------------------------------------------

    run_command(
        [
            str(python),
            str(
                CODE
                / "scripts"
                / "score_results.py"
            ),
            "--config",
            str(config_path),
        ],
        "BASE SCORING",
        CODE,
        env,
    )

    # --------------------------------------------------------------
    # STAGE 5: prompt-conditioned content scoring
    # --------------------------------------------------------------

    metrics_dir = (
        outputs / "metrics"
    )

    content_script = (
        CODE
        / "scripts"
        / "rescore_content.py"
    )

    if content_script.exists():

        run_command(
            [
                str(python),
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
            CODE,
            env,
        )

    else:

        print(
            "WARNING: "
            "rescore_content.py missing; "
            "content-aware scoring skipped."
        )

    update_state(
        state_path,
        state,
        "scoring",
        {
            "complete": True,
            "finished_at": now(),
            "metrics_dir":
                str(metrics_dir),
        },
    )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print(
        "\n"
        + "=" * 88
    )

    print(
        "FULL PLAN COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        f"Inference conditions : "
        f"{all_status['complete']}/"
        f"{all_status['expected']}"
    )

    print(
        f"Training checkpoints : "
        f"{tstatus['complete']}/"
        f"{tstatus['expected']}"
    )

    print(
        f"Raw outputs          : "
        f"{raw_dir}"
    )

    print(
        f"Parameter checkpoints: "
        f"{checkpoints}"
    )

    print(
        f"Metrics              : "
        f"{metrics_dir}"
    )

    print(
        f"State                : "
        f"{state_path}"
    )


if __name__ == "__main__":
    main()
