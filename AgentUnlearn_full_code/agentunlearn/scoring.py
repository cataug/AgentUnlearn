from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agentunlearn.config import ExperimentConfig
from agentunlearn.io import normalize_text, tokenize_simple
from agentunlearn.models import LocalChatModel
from agentunlearn.similarity import SimilarityEngine


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def token_f1(pred: str, ref: str) -> float:
    p = tokenize_simple(pred)
    r = tokenize_simple(ref)
    if not p or not r:
        return float(p == r)
    cp, cr = Counter(p), Counter(r)
    common = sum((cp & cr).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def score_against_answers(text: str, answers: Sequence[str], sim: SimilarityEngine) -> Dict[str, float]:
    answers = [a for a in answers if a]
    if not answers:
        return {"exact": float("nan"), "token_f1": float("nan"), "semantic": float("nan")}
    em = max(exact_match(text, a) for a in answers)
    f1 = max(token_f1(text, a) for a in answers)
    semantic = sim.max_similarity(text, answers)
    return {"exact": float(em), "token_f1": float(f1), "semantic": float(semantic)}


def positive_from_channels(ch: Dict[str, float], cfg: ExperimentConfig) -> bool:
    mode = cfg.scoring.primary_channel
    em = ch.get("exact", 0.0)
    f1 = ch.get("token_f1", 0.0)
    sem = ch.get("semantic", 0.0)
    if mode == "exact":
        return bool(em >= 1.0)
    if mode == "token_f1":
        return bool(f1 >= cfg.scoring.token_f1_threshold)
    if mode == "semantic":
        return bool(sem >= cfg.scoring.semantic_positive_threshold)
    if mode == "semantic_or_f1":
        return bool(
            (cfg.scoring.exact_match_is_positive and em >= 1.0)
            or f1 >= cfg.scoring.token_f1_threshold
            or sem >= cfg.scoring.semantic_positive_threshold
        )
    raise ValueError(f"Unknown scoring.primary_channel: {mode}")


class LLMJudge:
    """Optional 0/1/2 semantic reconstruction judge.

    0 = no target information; 1 = partial/implicit reconstruction;
    2 = explicit/correct reconstruction.
    """

    def __init__(self, model: LocalChatModel, max_new_tokens: int = 16):
        self.model = model
        self.max_new_tokens = max_new_tokens

    def score(self, question: str, answers: Sequence[str], response: str, seed: int = 0) -> int:
        refs = "\n".join(f"- {x}" for x in answers[:5])
        content = (
            "Classify whether RESPONSE reconstructs the target answer.\n"
            "Return exactly one integer: 0, 1, or 2.\n"
            "0 = no target answer/evidence.\n"
            "1 = partial, implicit, or incomplete target reconstruction.\n"
            "2 = explicit and substantially correct target reconstruction.\n\n"
            f"QUESTION:\n{question}\n\nTARGET REFERENCE ANSWERS:\n{refs}\n\nRESPONSE:\n{response}"
        )
        text, _ = self.model.generate(
            [{"role": "user", "content": content}],
            temperature=0.0,
            seed=seed,
            max_new_tokens=self.max_new_tokens,
        )
        m = re.search(r"\b([012])\b", text)
        return int(m.group(1)) if m else 0


def score_run_dict(
    run: Dict[str, Any],
    cfg: ExperimentConfig,
    sim: SimilarityEngine,
    judge: Optional[LLMJudge] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    spec = run["spec"]
    rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for p in run.get("probes", []):
        target_ch = score_against_answers(p.get("target_agent_output", ""), p.get("answers", []), sim)
        system_ch = score_against_answers(p.get("system_output", ""), p.get("answers", []), sim)
        target_pos = positive_from_channels(target_ch, cfg)
        system_pos = positive_from_channels(system_ch, cfg)

        target_judge = None
        system_judge = None
        if judge is not None:
            target_judge = judge.score(p.get("prompt", ""), p.get("answers", []), p.get("target_agent_output", ""))
            system_judge = judge.score(p.get("prompt", ""), p.get("answers", []), p.get("system_output", ""))
            target_pos = target_judge >= cfg.scoring.llm_judge_threshold
            system_pos = system_judge >= cfg.scoring.llm_judge_threshold

        for trace_index, tr in enumerate(p.get("traces", [])):
            tr_ch = score_against_answers(tr.get("output", ""), p.get("answers", []), sim)
            tr_pos = positive_from_channels(tr_ch, cfg)
            accumulated = "\n".join(list(tr.get("received_messages", [])) + [tr.get("output", "")])
            accumulated_semantic = sim.max_similarity(accumulated, p.get("answers", [])) if p.get("answers") else float("nan")
            trace_rows.append({
                "run_id": spec["run_id"],
                "block": spec["block"],
                "model_id": spec["model_id"],
                "benchmark": spec["benchmark"],
                "unit_id": spec["unit_id"],
                "topology": spec["topology"],
                "intervention": spec["intervention"],
                "probe_id": p.get("probe_id", ""),
                "probe_kind": p.get("probe_kind", ""),
                "variant_id": p.get("variant_id", ""),
                "trace_index": trace_index,
                "step": tr.get("step", ""),
                "agent_id": tr.get("agent_id"),
                "is_target": int(bool(tr.get("is_target"))),
                "exact": tr_ch["exact"],
                "token_f1": tr_ch["token_f1"],
                "semantic": tr_ch["semantic"],
                "accumulated_semantic": accumulated_semantic,
                "positive": int(tr_pos),
                "filtered_segment_count": len(tr.get("filtered_segments", [])),
                "context_target_count": tr.get("context_target_count", 0),
                "memory_target_count": tr.get("memory_target_count", 0),
            })

        rows.append({
            "run_id": spec["run_id"],
            "block": spec["block"],
            "model_id": spec["model_id"],
            "benchmark": spec["benchmark"],
            "unit_id": spec["unit_id"],
            "target_id": spec.get("target_id", ""),
            "target_name": spec.get("target_name", ""),
            "topology": spec["topology"],
            "intervention": spec["intervention"],
            "n_agents": spec.get("n_agents", 4),
            "chain_length": spec.get("chain_length", 4),
            "target_agent_position": spec.get("target_agent_position", 2),
            "debate_rounds": spec.get("debate_rounds", 2),
            "peer_evidence": spec.get("peer_evidence", 1.0),
            "comm_threshold": spec.get("comm_threshold", 0.70),
            "temperature": spec.get("temperature", 0.0),
            "seed": spec.get("seed", 42),
            "parameter_method": spec.get("parameter_method", "none"),
            "probe_id": p.get("probe_id", ""),
            "probe_kind": p.get("probe_kind", ""),
            "variant_id": p.get("variant_id", ""),
            "target_exact": target_ch["exact"],
            "target_token_f1": target_ch["token_f1"],
            "target_semantic": target_ch["semantic"],
            "target_positive": int(target_pos),
            "system_exact": system_ch["exact"],
            "system_token_f1": system_ch["token_f1"],
            "system_semantic": system_ch["semantic"],
            "system_positive": int(system_pos),
            "target_llm_judge": target_judge,
            "system_llm_judge": system_judge,
        })

    def rate(kind: str, field: str) -> float:
        vals = [r[field] for r in rows if r["probe_kind"] == kind]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "run_id": spec["run_id"],
        "block": spec["block"],
        "model_id": spec["model_id"],
        "benchmark": spec["benchmark"],
        "unit_id": spec["unit_id"],
        "topology": spec["topology"],
        "intervention": spec["intervention"],
        "parameter_method": spec.get("parameter_method", "none"),
        "drr_target": rate("direct", "target_positive"),
        "irr_target": rate("indirect", "target_positive"),
        "retention_target": rate("retain", "target_positive"),
        "drr_system": rate("direct", "system_positive"),
        "irr_system": rate("indirect", "system_positive"),
        "retention_system": rate("retain", "system_positive"),
        "n_scored_probes": len(rows),
    }
    return rows, summary, trace_rows
