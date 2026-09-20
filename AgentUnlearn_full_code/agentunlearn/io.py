from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np


def stable_int(*parts: Any, modulo: int = 2**31 - 1) -> int:
    raw = "||".join(str(x) for x in parts).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()
    return int(h[:16], 16) % modulo


def stable_sample(seq: Sequence[Any], n: int, *seed_parts: Any) -> List[Any]:
    items = list(seq)
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return items
    rng = random.Random(stable_int(*seed_parts))
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n]]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", normalize_text(text), flags=re.UNICODE)


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def recursive_dict_records(obj: Any) -> List[Dict[str, Any]]:
    """Extract record-like dictionaries from arbitrarily nested JSON.

    A dictionary is treated as a terminal record if it contains a likely text/QA
    field. Otherwise nested lists/dictionaries are recursively expanded.
    """
    text_keys = {
        "question", "prompt", "query", "input", "instruction", "text",
        "content", "answer", "response", "output", "completion", "choices",
    }
    out: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if text_keys.intersection(x.keys()):
                out.append(x)
                return
            emitted = False
            for v in x.values():
                if isinstance(v, (dict, list)):
                    emitted = True
                    walk(v)
            if not emitted:
                out.append(x)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return out


def first_text(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        if k in record and record[k] not in (None, ""):
            v = record[k]
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, (int, float, bool)):
                return str(v)
    return ""


def answer_list(record: Dict[str, Any]) -> List[str]:
    for k in ["answer", "answers", "response", "output", "completion", "gold", "target"]:
        if k not in record or record[k] in (None, ""):
            continue
        v = record[k]
        if isinstance(v, str):
            return [v.strip()]
        if isinstance(v, list):
            vals = []
            for z in v:
                if isinstance(z, str):
                    vals.append(z.strip())
                elif isinstance(z, dict):
                    t = first_text(z, ["text", "answer", "value"])
                    if t:
                        vals.append(t)
            if vals:
                return vals
        if isinstance(v, dict):
            t = first_text(v, ["text", "answer", "value"])
            if t:
                return [t]
    return []
