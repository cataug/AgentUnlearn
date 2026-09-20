from pathlib import Path
import csv

OUT = Path("/home/tahiti/AgentUnlearn/manifests")
OUT.mkdir(parents=True, exist_ok=True)

def save(name, rows):
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(path, len(rows))

models = [
    {
        "model_id": "qwen3_4b",
        "model_path": "/home/tahiti/AgentUnlearn/models/Qwen3-4B",
        "role": "main",
    },
    {
        "model_id": "qwen3_8b",
        "model_path": "/home/tahiti/AgentUnlearn/models/Qwen3-8B",
        "role": "main",
    },
    {
        "model_id": "mistral_7b",
        "model_path": "/home/tahiti/AgentUnlearn/models/Mistral-7B-Instruct-v0.3",
        "role": "main",
    },
    {
        "model_id": "qwen3_14b",
        "model_path": "/home/tahiti/AgentUnlearn/models/Qwen3-14B",
        "role": "scale_confirmation",
    },
    {
        "model_id": "qwen25coder_1p5b",
        "model_path": "/home/tahiti/AgentUnlearn/models/Qwen2.5-Coder-1.5B-Instruct",
        "role": "specialization_ablation",
    },
    {
        "model_id": "qwen25coder_7b",
        "model_path": "/home/tahiti/AgentUnlearn/models/Qwen2.5-Coder-7B-Instruct",
        "role": "specialization_ablation",
    },
    {
        "model_id": "qwen25coder_14b",
        "model_path": "/home/tahiti/AgentUnlearn/models/Qwen2.5-Coder-14B-Instruct",
        "role": "specialization_ablation",
    },
]

topologies = [
    {"topology_id": "independent",  "n_agents": 4},
    {"topology_id": "sequential",   "n_agents": 4},
    {"topology_id": "debate",       "n_agents": 4},
    {"topology_id": "scratchpad",   "n_agents": 4},
    {"topology_id": "hierarchical", "n_agents": 4},
]

interventions = [
    {
        "intervention_id": "none",
        "target_context": 0,
        "target_memory": 0,
        "communication_filter": 0,
        "parameter_unlearning": 0,
    },
    {
        "intervention_id": "global_context",
        "target_context": 1,
        "target_memory": 0,
        "communication_filter": 0,
        "parameter_unlearning": 0,
    },
    {
        "intervention_id": "target_context",
        "target_context": 1,
        "target_memory": 0,
        "communication_filter": 0,
        "parameter_unlearning": 0,
    },
    {
        "intervention_id": "target_memory",
        "target_context": 0,
        "target_memory": 1,
        "communication_filter": 0,
        "parameter_unlearning": 0,
    },
    {
        "intervention_id": "communication",
        "target_context": 0,
        "target_memory": 0,
        "communication_filter": 1,
        "parameter_unlearning": 0,
    },
    {
        "intervention_id": "target_context_comm",
        "target_context": 1,
        "target_memory": 0,
        "communication_filter": 1,
        "parameter_unlearning": 0,
    },
    {
        "intervention_id": "full_inference",
        "target_context": 1,
        "target_memory": 1,
        "communication_filter": 1,
        "parameter_unlearning": 0,
    },
]

save("models_manifest.csv", models)
save("topologies_manifest.csv", topologies)
save("interventions_manifest.csv", interventions)

print("\nDONE")
