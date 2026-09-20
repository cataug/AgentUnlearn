# Experiment semantics implemented by the code

## Unit of intervention

Agents are 1-indexed. `target_agent_position` identifies the designated agent whose forgetting/reconstruction behavior is the primary endpoint. Every agent uses the same base model within a condition; selectivity is induced by state/intervention differences, except parameter-level runs where the target condition uses a trained checkpoint.

## Knowledge pathways

Each benchmark bundle supplies target and retained evidence. Target/retained evidence is deterministically partitioned between visible context and private memory. Peer evidence is independently attenuated by `peer_evidence`.

- `none`: no channel suppression.
- `global_context`: target evidence is removed from visible context for every agent; private memory is unchanged.
- `target_context`: target evidence is removed only from the designated agent's visible context.
- `target_memory`: target evidence is removed only from the designated agent's private memory.
- `communication`: inbound peer messages to the designated agent are semantically filtered.
- `target_context_comm`: target context removal plus inbound communication filtering.
- `full_inference`: target context removal, target private-memory removal, and inbound communication filtering.
- `parameter_only`: trained parameter checkpoint, no communication filter.
- `parameter_plus_comm`: trained parameter checkpoint plus inbound communication filtering.

Communication filtering is sentence-level. A sentence is removed if its maximum similarity to the target reference set exceeds `comm_threshold`. The similarity backend is explicitly configured and logged.

## Topologies

- Independent: every agent answers without seeing peers; target output is also system output for the control.
- Sequential: agents act in order and see all previous responses.
- Debate: all agents first propose independently; then each debate round exposes peer proposals and requests a corrected answer.
- Scratchpad: each agent contributes to a shared note sequence; the last agent produces a final synthesis.
- Hierarchical: workers answer independently; the final meta-agent aggregates workers.

The last output produced by the designated agent is the target-agent endpoint. The topology's final aggregate/output is the system endpoint.

## Probe matching

Base probes are selected deterministically from each benchmark unit using the unit ID and probe kind. Consequently, matched intervention/topology comparisons use the same semantic test items. Five fixed surface forms are then applied to each selected base probe.

## Primary outcomes

DRR is the reconstruction-positive fraction on direct probes. IRR is the reconstruction-positive fraction on indirect probes. Retention is the positive fraction on retained-knowledge probes. All three are computed separately for the target agent and final system response.

Collateral Damage uses retained probes only and is calculated against the matched no-intervention baseline. Reconstruction Amplification compares IRR to Independent operation under the same model, benchmark unit, and intervention.

## Parameter baselines

`gradient_ascent` supports full-model or restricted trainable scopes. The configured scope is saved in checkpoint metadata. The default restricted scope is intentionally feasible on the local A100 and must be described as restricted gradient ascent in the manuscript unless changed to `all`.

`lora_grad_diff` optimizes `-L_forget + lambda * L_retain` using a PEFT LoRA adapter.
