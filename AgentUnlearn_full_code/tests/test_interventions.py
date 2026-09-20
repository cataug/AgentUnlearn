from agentunlearn.config import GenerationConfig
from agentunlearn.interventions import build_states
from agentunlearn.schemas import BenchmarkBundle, EvidenceItem, RunSpec


def spec(intervention):
    return RunSpec(
        run_id="x", block="main", model_id="m", benchmark="b", unit_id="u",
        target_id="t", target_name="T", topology="independent", intervention=intervention,
    )


def bundle():
    return BenchmarkBundle(
        benchmark="b", unit_id="u", target_id="t", target_name="T",
        direct_probes=[], indirect_probes=[], retain_probes=[],
        target_evidence=[EvidenceItem("t1", "target one", True), EvidenceItem("t2", "target two", True)],
        retain_evidence=[EvidenceItem("r1", "retain", False)],
    )


def test_target_context_removes_target_from_target_context():
    states = build_states(bundle(), spec("target_context"), GenerationConfig())
    target = [x for x in states if x.is_target][0]
    assert all(not x.is_target for x in target.context)


def test_full_inference_removes_target_context_and_memory():
    states = build_states(bundle(), spec("full_inference"), GenerationConfig())
    target = [x for x in states if x.is_target][0]
    assert all(not x.is_target for x in target.context)
    assert all(not x.is_target for x in target.memory)
