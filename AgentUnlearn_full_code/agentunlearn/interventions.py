from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from agentunlearn.config import GenerationConfig
from agentunlearn.io import stable_sample
from agentunlearn.schemas import (
    BenchmarkBundle,
    EvidenceItem,
    Probe,
    RunSpec,
)
from agentunlearn.similarity import SimilarityEngine


# ---------------------------------------------------------------------
# Full 2^3 inference-layer intervention design
#
# Context Memory Comm
#   0      0     0   none
#   1      0     0   target_context
#   0      1     0   target_memory
#   0      0     1   communication
#   1      1     0   target_context_memory
#   1      0     1   target_context_comm
#   0      1     1   target_memory_comm
#   1      1     1   full_inference
#
# global_context remains a separate baseline.
# ---------------------------------------------------------------------

_CONTEXT_REMOVAL = {
    "target_context",
    "target_context_memory",
    "target_context_comm",
    "full_inference",
}

_MEMORY_REMOVAL = {
    "target_memory",
    "target_context_memory",
    "target_memory_comm",
    "full_inference",
}

_COMM_FILTER = {
    "communication",
    "target_context_comm",
    "target_memory_comm",
    "full_inference",
    "parameter_plus_comm",
}


@dataclass
class AgentKnowledgeState:
    agent_id: int
    is_target: bool
    context: List[EvidenceItem]
    memory: List[EvidenceItem]


def _partition(
    items: Sequence[EvidenceItem],
    frac_context: float,
    frac_memory: float,
    *seed_parts,
) -> tuple[List[EvidenceItem], List[EvidenceItem]]:
    """
    Deterministically partition selected evidence between visible context
    and private memory.

    If context+memory fractions cover the whole evidence pool, ensure that
    small pools are not accidentally dropped because round(0.5) == 0.
    """
    items = list(items)

    if not items:
        return [], []

    ordered = stable_sample(
        items,
        len(items),
        *seed_parts,
    )

    n = len(items)

    n_ctx = min(
        n,
        max(
            0,
            int(round(n * frac_context)),
        ),
    )

    wanted_mem = max(
        0,
        int(round(n * frac_memory)),
    )

    # Important for n=1 and e.g. 0.5 / 0.5.
    if (
        n > 0
        and (frac_context + frac_memory) > 0
        and n_ctx + wanted_mem == 0
    ):
        if frac_context >= frac_memory:
            n_ctx = 1
        else:
            wanted_mem = 1

    # If the configured fractions jointly cover the evidence pool,
    # do not silently lose evidence due to integer rounding.
    if (
        (frac_context + frac_memory) >= 0.999
        and n_ctx + wanted_mem < n
    ):
        deficit = n - (n_ctx + wanted_mem)

        if frac_context >= frac_memory:
            n_ctx = min(
                n,
                n_ctx + deficit,
            )
        else:
            wanted_mem += deficit

    ctx = ordered[:n_ctx]

    remaining = ordered[n_ctx:]

    n_mem = min(
        len(remaining),
        wanted_mem,
    )

    mem = remaining[:n_mem]

    # If fractions intentionally sum to >1, overlap is permitted.
    if n_mem < wanted_mem:
        need = wanted_mem - n_mem
        mem += ordered[:need]

    return ctx, mem


def _norm_tokens(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            (text or "").lower(),
        )
    )


def _support_score(
    probe: Probe,
    evidence: EvidenceItem,
) -> tuple[float, int]:
    """
    Lightweight deterministic alignment between a probe and evidence.

    IMPORTANT:
    Gold answers are deliberately NOT used for evidence selection.
    This avoids answer-conditioned/oracle retrieval.

    The purpose here is only to prevent random evidence subsampling from
    dropping the most obviously relevant benchmark evidence.
    """
    q = _norm_tokens(probe.prompt)
    e = _norm_tokens(evidence.text)

    if not q or not e:
        return (0.0, 0)

    overlap = len(q & e)

    recall = overlap / max(
        1,
        len(q),
    )

    # tuple gives deterministic secondary preference
    # to larger absolute overlap.
    return (
        recall,
        overlap,
    )


def _sample_with_probe_support(
    items: Sequence[EvidenceItem],
    limit: int,
    probe: Probe | None,
    guarantee_support: bool,
    *seed_parts,
) -> List[EvidenceItem]:
    """
    Sample evidence while guaranteeing one question-aligned evidence item
    for the relevant evidence class.

    direct/indirect -> target evidence
    retain          -> retained evidence
    """
    items = list(items)

    limit = max(
        0,
        min(
            int(limit),
            len(items),
        ),
    )

    if limit == 0:
        return []

    if (
        not guarantee_support
        or probe is None
    ):
        return stable_sample(
            items,
            limit,
            *seed_parts,
        )

    support = max(
        items,
        key=lambda x: _support_score(
            probe,
            x,
        ),
    )

    remaining = [
        x
        for x in items
        if x.evidence_id != support.evidence_id
    ]

    sampled = stable_sample(
        remaining,
        max(
            0,
            limit - 1,
        ),
        *seed_parts,
        "rest",
    )

    return [support] + sampled


def build_states(
    bundle: BenchmarkBundle,
    spec: RunSpec,
    cfg: GenerationConfig,
    probe: Probe | None = None,
) -> List[AgentKnowledgeState]:
    """
    Construct per-agent knowledge states for one probe.

    Probe-aware evidence construction ensures:

    - direct/indirect probes retain one question-aligned target item
      BEFORE the intervention is applied;
    - retain probes retain one question-aligned retained item;
    - the intervention then determines which target-agent pathways
      remain available.

    This separates forgetting/reconstruction measurement from accidental
    evidence-subsampling failure.
    """

    target_items = _sample_with_probe_support(
        bundle.target_evidence,
        cfg.max_target_evidence,
        probe,
        bool(
            probe
            and probe.kind in {
                "direct",
                "indirect",
            }
        ),
        bundle.unit_id,
        probe.probe_id if probe else "no-probe",
        "target-evidence",
    )

    retain_items = _sample_with_probe_support(
        bundle.retain_evidence,
        cfg.max_retain_evidence,
        probe,
        bool(
            probe
            and probe.kind == "retain"
        ),
        bundle.unit_id,
        probe.probe_id if probe else "no-probe",
        "retain-evidence",
    )

    target_ctx, target_mem = _partition(
        target_items,
        cfg.context_target_fraction,
        cfg.memory_target_fraction,
        bundle.unit_id,
        probe.probe_id if probe else "no-probe",
        "target-partition",
    )

    retain_ctx, retain_mem = _partition(
        retain_items,
        cfg.context_retain_fraction,
        cfg.memory_retain_fraction,
        bundle.unit_id,
        probe.probe_id if probe else "no-probe",
        "retain-partition",
    )

    # Fraction of target evidence available to non-target peer agents.
    peer_target_ctx = stable_sample(
        target_ctx,
        int(
            round(
                len(target_ctx)
                * spec.peer_evidence
            )
        ),
        bundle.unit_id,
        probe.probe_id if probe else "no-probe",
        spec.peer_evidence,
        "peer-ctx",
    )

    peer_target_mem = stable_sample(
        target_mem,
        int(
            round(
                len(target_mem)
                * spec.peer_evidence
            )
        ),
        bundle.unit_id,
        probe.probe_id if probe else "no-probe",
        spec.peer_evidence,
        "peer-mem",
    )

    states: List[AgentKnowledgeState] = []

    target_pos = max(
        1,
        min(
            spec.target_agent_position,
            max(
                1,
                spec.n_agents,
            ),
        ),
    )

    for aid in range(
        1,
        spec.n_agents + 1,
    ):
        is_target = (
            aid == target_pos
        )

        ctx = list(
            (
                target_ctx
                if is_target
                else peer_target_ctx
            )
            + retain_ctx
        )

        mem = list(
            (
                target_mem
                if is_target
                else peer_target_mem
            )
            + retain_mem
        )

        # -------------------------------------------------------------
        # Context intervention
        # -------------------------------------------------------------

        if spec.intervention == "global_context":
            # Context-only baseline applied uniformly to every agent.
            ctx = [
                x
                for x in ctx
                if not x.is_target
            ]

        elif (
            is_target
            and spec.intervention
            in _CONTEXT_REMOVAL
        ):
            ctx = [
                x
                for x in ctx
                if not x.is_target
            ]

        # -------------------------------------------------------------
        # Private-memory intervention
        # -------------------------------------------------------------

        if (
            is_target
            and spec.intervention
            in _MEMORY_REMOVAL
        ):
            mem = [
                x
                for x in mem
                if not x.is_target
            ]

        states.append(
            AgentKnowledgeState(
                agent_id=aid,
                is_target=is_target,
                context=ctx,
                memory=mem,
            )
        )

    return states


def communication_filter_enabled(
    spec: RunSpec,
) -> bool:
    return (
        spec.intervention
        in _COMM_FILTER
    )


def maybe_filter_inbound(
    message: str,
    receiver_is_target: bool,
    spec: RunSpec,
    similarity: SimilarityEngine,
    target_references: Sequence[str],
) -> tuple[str, List[dict]]:
    """
    Communication filtering is applied only to messages delivered to
    the designated target agent.

    Peer agents themselves remain unchanged.
    """
    if (
        not receiver_is_target
        or not communication_filter_enabled(spec)
    ):
        return message, []

    return similarity.filter_text(
        message,
        target_references,
        spec.comm_threshold,
    )
