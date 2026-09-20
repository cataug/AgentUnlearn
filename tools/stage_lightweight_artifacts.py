#!/usr/bin/env python3

from pathlib import Path
import subprocess


ROOT = Path("/home/tahiti/AgentUnlearn")

MAX_BYTES = 20 * 1024 * 1024  # 20 MiB


# These trees are explicitly wanted in Git.
INCLUDE_ROOTS = [
    ROOT / "AgentUnlearn_full_code",
    ROOT / "scripts",
    ROOT / "config",
    ROOT / "manifests",
    ROOT / "STATS",
    ROOT / "FIGURES",
    ROOT / "TECH_STATS",
    ROOT / "outputs" / "full_plan" / "metrics",
]


# Additional useful top-level documentation/code files.
TOP_LEVEL_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".tex",
    ".bib",
    ".sh",
}


EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",

    "data",
    "datasets",
    "models",
    "checkpoints",
    "raw",

    ".cache",
    "huggingface",
    "wandb",
}


EXCLUDED_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
}


def excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)

    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True

    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True

    return False


def candidate_files():
    found = set()

    for root in INCLUDE_ROOTS:
        if not root.exists():
            continue

        if root.is_file():
            found.add(root)
            continue

        for p in root.rglob("*"):
            if p.is_file():
                found.add(p)

    # Useful lightweight files sitting directly in project root.
    for p in ROOT.iterdir():
        if (
            p.is_file()
            and (
                p.suffix.lower() in TOP_LEVEL_SUFFIXES
                or p.name in {
                    ".gitignore",
                    "LICENSE",
                    "Makefile",
                    "requirements.txt",
                    "pyproject.toml",
                }
            )
        ):
            found.add(p)

    # Include this helper itself.
    helper = ROOT / "tools" / "stage_lightweight_artifacts.py"
    if helper.exists():
        found.add(helper)

    return sorted(found)


accepted = []
too_large = []
ignored = []


for p in candidate_files():

    if excluded(p):
        ignored.append(p)
        continue

    try:
        size = p.stat().st_size
    except OSError:
        continue

    if size > MAX_BYTES:
        too_large.append(
            (p, size)
        )
        continue

    accepted.append(p)


print("=" * 90)
print("LIGHTWEIGHT GIT STAGING")
print("=" * 90)

print(f"Accepted : {len(accepted)}")
print(f"Too large: {len(too_large)}")
print(f"Excluded : {len(ignored)}")


if too_large:
    print("\nFILES SKIPPED BECAUSE > 20 MiB")

    for p, size in sorted(
        too_large,
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"{size / 1024**2:8.1f} MiB  "
            f"{p.relative_to(ROOT)}"
        )


# git add in batches to avoid huge argv.
BATCH = 100

for start in range(
    0,
    len(accepted),
    BATCH,
):
    batch = accepted[
        start:start + BATCH
    ]

    rels = [
        str(p.relative_to(ROOT))
        for p in batch
    ]

    subprocess.run(
        [
            "git",
            "add",
            "--",
            *rels,
        ],
        cwd=ROOT,
        check=True,
    )


print()
print("Staging complete.")
print("No datasets/models/raw/checkpoints were added.")
