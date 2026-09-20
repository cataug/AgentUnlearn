from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Probe:
    probe_id: str
    prompt: str
    answers: List[str]
    kind: str  # direct | indirect | retain
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    evidence_id: str
    text: str
    is_target: bool
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkBundle:
    benchmark: str
    unit_id: str
    target_id: str
    target_name: str
    direct_probes: List[Probe]
    indirect_probes: List[Probe]
    retain_probes: List[Probe]
    target_evidence: List[EvidenceItem]
    retain_evidence: List[EvidenceItem]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSpec:
    run_id: str
    block: str
    model_id: str
    benchmark: str
    unit_id: str
    target_id: str
    target_name: str
    topology: str
    intervention: str
    n_agents: int = 4
    chain_length: int = 4
    target_agent_position: int = 2
    debate_rounds: int = 2
    peer_evidence: float = 1.0
    comm_threshold: float = 0.70
    temperature: float = 0.0
    seed: int = 42
    parameter_method: str = "none"
    status: str = "pending"

    @classmethod
    def from_mapping(cls, row: Dict[str, Any]) -> "RunSpec":
        def _i(k: str, default: int) -> int:
            v = row.get(k, default)
            if v in (None, ""):
                return default
            return int(float(v))

        def _f(k: str, default: float) -> float:
            v = row.get(k, default)
            if v in (None, ""):
                return default
            return float(v)

        return cls(
            run_id=str(row["run_id"]),
            block=str(row.get("block", "main")),
            model_id=str(row["model_id"]),
            benchmark=str(row["benchmark"]),
            unit_id=str(row["unit_id"]),
            target_id=str(row.get("target_id", "")),
            target_name=str(row.get("target_name", "")),
            topology=str(row["topology"]),
            intervention=str(row["intervention"]),
            n_agents=_i("n_agents", 4),
            chain_length=_i("chain_length", 4),
            target_agent_position=_i("target_agent_position", 2),
            debate_rounds=_i("debate_rounds", 2),
            peer_evidence=_f("peer_evidence", 1.0),
            comm_threshold=_f("comm_threshold", 0.70),
            temperature=_f("temperature", 0.0),
            seed=_i("seed", 42),
            parameter_method=str(row.get("parameter_method", "none")),
            status=str(row.get("status", "pending")),
        )


@dataclass
class AgentCallTrace:
    agent_id: int
    is_target: bool
    step: str
    prompt_messages: List[Dict[str, str]]
    output: str
    received_messages: List[str] = field(default_factory=list)
    filtered_segments: List[Dict[str, Any]] = field(default_factory=list)
    context_target_count: int = 0
    context_retain_count: int = 0
    memory_target_count: int = 0
    memory_retain_count: int = 0
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    probe_id: str
    probe_kind: str
    variant_id: str
    prompt: str
    answers: List[str]
    target_agent_output: str
    system_output: str
    traces: List[AgentCallTrace]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    spec: RunSpec
    benchmark_metadata: Dict[str, Any]
    probes: List[ProbeResult]
    started_at: str
    finished_at: str
    elapsed_sec: float
    model_metadata: Dict[str, Any]
    status: str = "complete"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
