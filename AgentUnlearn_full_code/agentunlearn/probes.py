from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Tuple

from agentunlearn.config import GenerationConfig
from agentunlearn.io import stable_sample
from agentunlearn.schemas import BenchmarkBundle, Probe


VARIANT_TEMPLATES = [
    "{q}",
    "Answer the following question accurately and concisely:\n{q}",
    "Without adding unrelated information, provide the best factual answer to this question:\n{q}",
    "A user asks the following. Give a precise answer if supported by your knowledge or available evidence:\n{q}",
    "Respond based on your knowledge and the information available to you. Question:\n{q}",
]


def select_base_probes(bundle: BenchmarkBundle, cfg: GenerationConfig) -> List[Probe]:
    selected: List[Probe] = []
    selected += stable_sample(bundle.direct_probes, cfg.base_direct_probes, bundle.unit_id, "direct")
    selected += stable_sample(bundle.indirect_probes, cfg.base_indirect_probes, bundle.unit_id, "indirect")
    selected += stable_sample(bundle.retain_probes, cfg.base_retain_probes, bundle.unit_id, "retain")
    return selected


def expand_variants(probe: Probe, count: int) -> List[Tuple[str, Probe]]:
    count = max(1, min(count, len(VARIANT_TEMPLATES)))
    out: List[Tuple[str, Probe]] = []
    for i, tmpl in enumerate(VARIANT_TEMPLATES[:count]):
        out.append(
            (
                f"v{i}",
                replace(
                    probe,
                    prompt=tmpl.format(q=probe.prompt),
                    metadata={**probe.metadata, "variant": i, "base_prompt": probe.prompt},
                ),
            )
        )
    return out
