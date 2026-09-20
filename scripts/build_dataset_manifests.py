from pathlib import Path
import csv
import json
import re
from collections import Counter, defaultdict

ROOT = Path("/home/tahiti/AgentUnlearn")
OUT = ROOT / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "TOFU": ROOT / "data/TOFU",
    "MUSE-News": ROOT / "data/MUSE-News",
    "MUSE-Books": ROOT / "data/MUSE-Books",
    "RWKU": ROOT / "data/RWKU",
    "WMDP": ROOT / "data/WMDP",
}

SUPPORTED = {".json", ".jsonl", ".csv", ".parquet"}

PROMPT_KEYS = [
    "question", "prompt", "query", "input",
    "instruction", "text", "forget_prompt"
]

ANSWER_KEYS = [
    "answer", "response", "output", "target",
    "completion", "gold", "label", "forget_answer"
]


def first_value(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            v = d[k]
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return str(v)
    return ""


def flatten_scalar_fields(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
    return out


def infer_role(text):
    s = text.lower()

    if any(x in s for x in [
        "forget", "unlearn", "target", "erase"
    ]):
        return "forget"

    if any(x in s for x in [
        "retain", "neighbor", "utility", "bbh"
    ]):
        return "retain"

    if any(x in s for x in [
        "mia", "truthful", "attack", "adversarial"
    ]):
        return "evaluation"

    return "unspecified"


def infer_split(rec, rel):
    for k in [
        "split", "subset", "partition", "config",
        "task", "type", "category"
    ]:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k])

    return rel.stem


def infer_rwku_target(rel):
    parts = rel.parts

    if "Target" not in parts:
        return "", ""

    i = parts.index("Target")

    if i + 1 >= len(parts):
        return "", ""

    folder = parts[i + 1]

    m = re.match(r"^(\d+)_(.+)$", folder)

    if not m:
        return folder, folder.replace("_", " ")

    tid = m.group(1)
    name = m.group(2).replace("_", " ")

    return tid, name


def read_records(path):
    ext = path.suffix.lower()

    if ext == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        records.append({"value": obj})
                except Exception:
                    records.append({"_parse_error": line[:500]})
        return records

    if ext == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f))

    if ext == ".json":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return [
                x if isinstance(x, dict) else {"value": x}
                for x in obj
            ]

        if isinstance(obj, dict):
            # Common pattern: {"data": [...]}
            candidate_lists = [
                (k, v) for k, v in obj.items()
                if isinstance(v, list) and v
            ]

            if len(candidate_lists) == 1:
                k, vals = candidate_lists[0]
                return [
                    x if isinstance(x, dict)
                    else {"value": x, "_container": k}
                    for x in vals
                ]

            return [obj]

        return [{"value": obj}]

    if ext == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        except Exception as e:
            return [{"_parquet_error": repr(e)}]

    return []


files_rows = []
item_rows = []
schema_counter = defaultdict(Counter)
target_counter = defaultdict(Counter)

for benchmark, root in DATASETS.items():

    print("\n" + "=" * 90)
    print(benchmark, "->", root)

    if not root.exists():
        print("MISSING")
        continue

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    ]

    print("candidate files:", len(files))

    for path in files:
        rel = path.relative_to(root)

        try:
            records = read_records(path)
            parse_status = "ok"
        except Exception as e:
            records = []
            parse_status = f"error:{type(e).__name__}"

        all_keys = Counter()

        for rec in records:
            if not isinstance(rec, dict):
                rec = {"value": rec}

            for k in rec.keys():
                all_keys[k] += 1
                schema_counter[benchmark][k] += 1

        files_rows.append({
            "benchmark": benchmark,
            "relative_path": str(rel),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "record_count": len(records),
            "parse_status": parse_status,
            "keys": "|".join(sorted(all_keys.keys())),
        })

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                rec = {"value": rec}

            split = infer_split(rec, rel)

            path_context = " ".join([
                str(rel),
                split,
                first_value(rec, ["type", "category", "task"])
            ])

            role = infer_role(path_context)

            target_id = ""
            target_name = ""

            if benchmark == "RWKU":
                target_id, target_name = infer_rwku_target(rel)

            # Try common target/entity fields for other datasets.
            if not target_id:
                target_id = first_value(rec, [
                    "target_id", "forget_id", "author_id",
                    "entity_id", "id"
                ])

            if not target_name:
                target_name = first_value(rec, [
                    "target_name", "entity", "author",
                    "name", "subject"
                ])

            prompt = first_value(rec, PROMPT_KEYS)
            answer = first_value(rec, ANSWER_KEYS)

            probe_type = rel.stem

            item_id = f"{benchmark}:{rel}:{idx}"

            scalar = flatten_scalar_fields(rec)

            item_rows.append({
                "benchmark": benchmark,
                "domain": (
                    "news" if benchmark == "MUSE-News"
                    else "books" if benchmark == "MUSE-Books"
                    else ""
                ),
                "split": split,
                "role": role,
                "target_id": target_id,
                "target_name": target_name,
                "probe_type": probe_type,
                "item_id": item_id,
                "prompt": prompt,
                "answer": answer,
                "source_file": str(rel),
                "source_row": idx,
                "original_fields": json.dumps(
                    scalar,
                    ensure_ascii=False,
                    sort_keys=True
                ),
            })

            if target_id or target_name:
                key = (target_id, target_name, role, probe_type)
                target_counter[benchmark][key] += 1


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


write_csv(
    OUT / "dataset_files_manifest.csv",
    files_rows,
    [
        "benchmark",
        "relative_path",
        "extension",
        "size_bytes",
        "record_count",
        "parse_status",
        "keys",
    ],
)

write_csv(
    OUT / "dataset_items_manifest.csv",
    item_rows,
    [
        "benchmark",
        "domain",
        "split",
        "role",
        "target_id",
        "target_name",
        "probe_type",
        "item_id",
        "prompt",
        "answer",
        "source_file",
        "source_row",
        "original_fields",
    ],
)

target_rows = []

for benchmark, counts in target_counter.items():
    aggregate = defaultdict(lambda: {
        "forget": 0,
        "retain": 0,
        "evaluation": 0,
        "unspecified": 0,
        "probe_types": set(),
    })

    for (tid, name, role, probe), n in counts.items():
        key = (tid, name)
        aggregate[key][role] += n
        aggregate[key]["probe_types"].add(probe)

    for (tid, name), stats in aggregate.items():
        target_rows.append({
            "benchmark": benchmark,
            "target_id": tid,
            "target_name": name,
            "forget_items": stats["forget"],
            "retain_items": stats["retain"],
            "evaluation_items": stats["evaluation"],
            "unspecified_items": stats["unspecified"],
            "probe_types": "|".join(sorted(stats["probe_types"])),
        })

write_csv(
    OUT / "targets_manifest.csv",
    target_rows,
    [
        "benchmark",
        "target_id",
        "target_name",
        "forget_items",
        "retain_items",
        "evaluation_items",
        "unspecified_items",
        "probe_types",
    ],
)

schema_rows = []

for benchmark, counter in schema_counter.items():
    for key, count in counter.most_common():
        schema_rows.append({
            "benchmark": benchmark,
            "field": key,
            "occurrences": count,
        })

write_csv(
    OUT / "schema_manifest.csv",
    schema_rows,
    ["benchmark", "field", "occurrences"],
)

print("\n" + "=" * 90)
print("DONE")
print("files :", len(files_rows))
print("items :", len(item_rows))
print("targets:", len(target_rows))

print("\nOutputs:")
for name in [
    "dataset_files_manifest.csv",
    "dataset_items_manifest.csv",
    "targets_manifest.csv",
    "schema_manifest.csv",
]:
    print(" ", OUT / name)
