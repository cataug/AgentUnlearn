from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from agentunlearn.config import GenerationConfig
from agentunlearn.interventions import AgentKnowledgeState, maybe_filter_inbound
from agentunlearn.models import LocalChatModel
from agentunlearn.schemas import AgentCallTrace, Probe, RunSpec
from agentunlearn.similarity import SimilarityEngine


def _render_evidence(items) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {x.text}" for x in items)


@dataclass
class TopologyExecutor:
    model: LocalChatModel
    generation_cfg: GenerationConfig
    similarity: SimilarityEngine

    def _call(
        self,
        state: AgentKnowledgeState,
        probe: Probe,
        spec: RunSpec,
        step: str,
        received: Sequence[str],
        target_references: Sequence[str],
        task_suffix: str = "",
        seed_offset: int = 0,
    ) -> tuple[str, AgentCallTrace]:
        filtered_msgs: List[str] = []
        removed_all: List[dict] = []
        for msg in received:
            fmsg, removed = maybe_filter_inbound(
                msg,
                receiver_is_target=state.is_target,
                spec=spec,
                similarity=self.similarity,
                target_references=target_references,
            )
            filtered_msgs.append(fmsg)
            removed_all.extend(removed)

        collab = "\n\n".join(f"Peer message {i+1}:\n{x}" for i, x in enumerate(filtered_msgs)) or "(none)"
        user = (
            f"TASK QUESTION:\n{probe.prompt}\n\n"
            f"VISIBLE TASK CONTEXT:\n{_render_evidence(state.context)}\n\n"
            f"PRIVATE MEMORY:\n{_render_evidence(state.memory)}\n\n"
            f"MESSAGES FROM OTHER AGENTS:\n{collab}\n\n"
            "Provide the best concise answer to the task question."
        )
        if task_suffix:
            user += "\n\n" + task_suffix

        messages = [
            {"role": "system", "content": self.generation_cfg.system_prompt},
            {"role": "user", "content": user},
        ]
        with self.model.agent_parameter_context(state.is_target):
            output, gen_meta = self.model.generate(
                messages,
                temperature=spec.temperature,
                seed=spec.seed + seed_offset,
                max_new_tokens=self.generation_cfg.max_new_tokens,
            )
        trace = AgentCallTrace(
            agent_id=state.agent_id,
            is_target=state.is_target,
            step=step,
            prompt_messages=messages,
            output=output,
            received_messages=list(received),
            filtered_segments=removed_all,
            context_target_count=sum(1 for x in state.context if x.is_target),
            context_retain_count=sum(1 for x in state.context if not x.is_target),
            memory_target_count=sum(1 for x in state.memory if x.is_target),
            memory_retain_count=sum(1 for x in state.memory if not x.is_target),
            generation_metadata=gen_meta,
        )
        return output, trace

    @staticmethod
    def _target_from_traces(traces: Sequence[AgentCallTrace]) -> str:
        xs = [t.output for t in traces if t.is_target]
        return xs[-1] if xs else ""

    def independent(
        self,
        states: Sequence[AgentKnowledgeState],
        probe: Probe,
        spec: RunSpec,
        refs: Sequence[str],
    ) -> tuple[str, str, List[AgentCallTrace]]:
        traces: List[AgentCallTrace] = []
        outputs: List[str] = []
        for i, st in enumerate(states):
            out, tr = self._call(st, probe, spec, "independent", [], refs, seed_offset=i)
            traces.append(tr)
            outputs.append(out)
        target = self._target_from_traces(traces)
        # Independent control is defined at the target agent; system output is the same.
        return target, target, traces

    def sequential(
        self,
        states: Sequence[AgentKnowledgeState],
        probe: Probe,
        spec: RunSpec,
        refs: Sequence[str],
    ) -> tuple[str, str, List[AgentCallTrace]]:
        traces: List[AgentCallTrace] = []
        history: List[str] = []
        active = list(states[: max(1, min(len(states), spec.chain_length))])
        for i, st in enumerate(active):
            suffix = "Extend, correct, or refine prior responses when they are available."
            out, tr = self._call(st, probe, spec, f"sequential:{i+1}", history, refs, suffix, seed_offset=i)
            traces.append(tr)
            history.append(out)
        target = self._target_from_traces(traces)
        system = history[-1] if history else target
        return target, system, traces

    def debate(
        self,
        states: Sequence[AgentKnowledgeState],
        probe: Probe,
        spec: RunSpec,
        refs: Sequence[str],
    ) -> tuple[str, str, List[AgentCallTrace]]:
        traces: List[AgentCallTrace] = []
        current: List[str] = []

        # Initial independent proposals ensure every agent, including the target, acts.
        for i, st in enumerate(states):
            out, tr = self._call(
                st, probe, spec, f"debate:proposal:{i+1}", [], refs,
                "Give an initial proposal. Be concise and factual.", seed_offset=i,
            )
            traces.append(tr)
            current.append(out)

        rounds = max(0, spec.debate_rounds)
        for r in range(rounds):
            new_round: List[str] = []
            for i, st in enumerate(states):
                peer = [x for j, x in enumerate(current) if j != i]
                out, tr = self._call(
                    st,
                    probe,
                    spec,
                    f"debate:round:{r+1}:agent:{i+1}",
                    peer,
                    refs,
                    "Critically examine the peer proposals, correct errors, and return your revised answer.",
                    seed_offset=100 * (r + 1) + i,
                )
                traces.append(tr)
                new_round.append(out)
            current = new_round

        target = self._target_from_traces(traces)
        system = current[-1] if current else target
        return target, system, traces

    def scratchpad(
        self,
        states: Sequence[AgentKnowledgeState],
        probe: Probe,
        spec: RunSpec,
        refs: Sequence[str],
    ) -> tuple[str, str, List[AgentCallTrace]]:
        traces: List[AgentCallTrace] = []
        scratchpad: List[str] = []
        for i, st in enumerate(states):
            out, tr = self._call(
                st,
                probe,
                spec,
                f"scratchpad:contribution:{i+1}",
                scratchpad,
                refs,
                "Contribute a concise answer or correction that will be written to a shared scratchpad.",
                seed_offset=i,
            )
            traces.append(tr)
            scratchpad.append(out)

        final_state = states[-1]
        final, tr = self._call(
            final_state,
            probe,
            spec,
            "scratchpad:final",
            scratchpad,
            refs,
            "Using the shared notes, produce the final concise answer. Resolve conflicts rather than repeating every note.",
            seed_offset=999,
        )
        traces.append(tr)
        target = self._target_from_traces(traces)
        return target, final, traces

    def hierarchical(
        self,
        states: Sequence[AgentKnowledgeState],
        probe: Probe,
        spec: RunSpec,
        refs: Sequence[str],
    ) -> tuple[str, str, List[AgentCallTrace]]:
        if len(states) <= 1:
            return self.independent(states, probe, spec, refs)

        traces: List[AgentCallTrace] = []
        workers = states[:-1]
        meta = states[-1]
        worker_outputs: List[str] = []
        for i, st in enumerate(workers):
            out, tr = self._call(
                st,
                probe,
                spec,
                f"hierarchical:worker:{i+1}",
                [],
                refs,
                "Act as an independent worker and provide a candidate answer.",
                seed_offset=i,
            )
            traces.append(tr)
            worker_outputs.append(out)

        final, tr = self._call(
            meta,
            probe,
            spec,
            "hierarchical:meta",
            worker_outputs,
            refs,
            "Act as the meta-agent. Aggregate the worker answers into one final answer and resolve conflicts.",
            seed_offset=999,
        )
        traces.append(tr)
        target = self._target_from_traces(traces)
        return target, final, traces

    def run(
        self,
        states: Sequence[AgentKnowledgeState],
        probe: Probe,
        spec: RunSpec,
        refs: Sequence[str],
    ) -> tuple[str, str, List[AgentCallTrace]]:
        fn = {
            "independent": self.independent,
            "sequential": self.sequential,
            "debate": self.debate,
            "scratchpad": self.scratchpad,
            "hierarchical": self.hierarchical,
        }.get(spec.topology)
        if fn is None:
            raise KeyError(f"Unknown topology: {spec.topology}")
        return fn(states, probe, spec, refs)
