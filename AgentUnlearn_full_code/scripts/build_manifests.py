#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import csv
from pathlib import Path

ROOT = Path("/home/tahiti/AgentUnlearn")
OUT = ROOT / "manifests"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, rows):
    p = OUT / name
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(name, len(rows))


models = [
    {"model_id":"qwen3_4b","model_path":str(ROOT/"models/Qwen3-4B"),"role":"main"},
    {"model_id":"qwen3_8b","model_path":str(ROOT/"models/Qwen3-8B"),"role":"main"},
    {"model_id":"mistral_7b","model_path":str(ROOT/"models/Mistral-7B-Instruct-v0.3"),"role":"main"},
    {"model_id":"qwen3_14b","model_path":str(ROOT/"models/Qwen3-14B"),"role":"large_scale"},
    {"model_id":"qwen25coder_1p5b","model_path":str(ROOT/"models/Qwen2.5-Coder-1.5B-Instruct"),"role":"coder_ablation"},
    {"model_id":"qwen25coder_7b","model_path":str(ROOT/"models/Qwen2.5-Coder-7B-Instruct"),"role":"coder_ablation"},
    {"model_id":"qwen25coder_14b","model_path":str(ROOT/"models/Qwen2.5-Coder-14B-Instruct"),"role":"coder_ablation"},
]
save("models_manifest.csv", models)

units = [
    {"unit_id":"tofu_forget01","benchmark":"TOFU","target_id":"forget01","target_name":"forget01","domain":"synthetic_factual","tier":"main"},
    {"unit_id":"tofu_forget05","benchmark":"TOFU","target_id":"forget05","target_name":"forget05","domain":"synthetic_factual","tier":"main"},
    {"unit_id":"tofu_forget10","benchmark":"TOFU","target_id":"forget10","target_name":"forget10","domain":"synthetic_factual","tier":"main"},
    {"unit_id":"muse_news","benchmark":"MUSE-News","target_id":"news","target_name":"MUSE News","domain":"news","tier":"main"},
    {"unit_id":"muse_books","benchmark":"MUSE-Books","target_id":"books","target_name":"MUSE Books","domain":"books","tier":"main"},
]

rwku_root = ROOT / "data/RWKU/Target"
rwku = []
if rwku_root.exists():
    for p in rwku_root.iterdir():
        if not p.is_dir() or "_" not in p.name:
            continue
        a, b = p.name.split("_", 1)
        try:
            rwku.append((int(a), b.replace("_", " ")))
        except ValueError:
            pass
rwku.sort()
if len(rwku) >= 10:
    idxs = [round(i*(len(rwku)-1)/9) for i in range(10)]
    rwku = [rwku[i] for i in idxs]
for tid, name in rwku:
    units.append({"unit_id":f"rwku_{tid:03d}","benchmark":"RWKU","target_id":str(tid),"target_name":name,"domain":"real_world_entity","tier":"main"})
for subset in ["bio","cyber"]:
    units.append({"unit_id":f"wmdp_{subset}","benchmark":"WMDP","target_id":subset,"target_name":f"WMDP {subset}","domain":"safety","tier":"appendix"})
save("benchmark_units_manifest.csv", units)

main_units = [u for u in units if u["tier"] == "main"]
appendix_units = [u for u in units if u["tier"] == "appendix"]

TOPO = ["independent","sequential","debate","scratchpad","hierarchical"]
INTERV = ["none","global_context","target_context","target_memory","communication","target_context_comm","full_inference"]
save("topologies_manifest.csv", [{"topology_id":x,"n_agents":4} for x in TOPO])
save("interventions_manifest.csv", [
    {"intervention_id":"none","target_context":0,"target_memory":0,"communication_filter":0,"parameter_unlearning":0},
    {"intervention_id":"global_context","target_context":1,"target_memory":0,"communication_filter":0,"parameter_unlearning":0},
    {"intervention_id":"target_context","target_context":1,"target_memory":0,"communication_filter":0,"parameter_unlearning":0},
    {"intervention_id":"target_memory","target_context":0,"target_memory":1,"communication_filter":0,"parameter_unlearning":0},
    {"intervention_id":"communication","target_context":0,"target_memory":0,"communication_filter":1,"parameter_unlearning":0},
    {"intervention_id":"target_context_comm","target_context":1,"target_memory":0,"communication_filter":1,"parameter_unlearning":0},
    {"intervention_id":"full_inference","target_context":1,"target_memory":1,"communication_filter":1,"parameter_unlearning":0},
])
KEY = ["none","target_context","target_context_comm"]
ABINT = ["target_context","target_context_comm"]
MAIN_MODELS = ["qwen3_4b","qwen3_8b","mistral_7b"]
CODER_MODELS = ["qwen25coder_1p5b","qwen25coder_7b","qwen25coder_14b"]
PARAM_MODELS = ["qwen3_4b","qwen3_8b"]

counter=0
def run(block, model, unit, topology, intervention, **kw):
    global counter
    counter += 1
    x = dict(
        run_id=f"AU{counter:06d}", block=block, model_id=model,
        benchmark=unit["benchmark"], unit_id=unit["unit_id"], target_id=unit["target_id"], target_name=unit["target_name"],
        topology=topology, intervention=intervention, n_agents=4, chain_length=4,
        target_agent_position=2, debate_rounds=2, peer_evidence=1.0, comm_threshold=0.70,
        temperature=0.0, seed=42, parameter_method="none", status="pending",
    )
    x.update(kw); return x

runs_main=[run("main",m,u,t,i) for m in MAIN_MODELS for u in main_units for t in TOPO for i in INTERV]
runs_large=[run("large_scale","qwen3_14b",u,t,i) for u in main_units for t in TOPO for i in KEY]
runs_coder=[run("coder_scale",m,u,t,i) for m in CODER_MODELS for u in main_units for t in ["independent","debate","scratchpad"] for i in KEY]

wanted={"tofu_forget01","tofu_forget05","tofu_forget10","muse_news"}
ab_units=[u for u in main_units if u["unit_id"] in wanted]
rwku_one=next((u for u in main_units if u["benchmark"]=="RWKU"),None)
if rwku_one: ab_units.append(rwku_one)

abl=[]
for u in ab_units:
    for t in ["sequential","debate","scratchpad"]:
        for i in ABINT:
            for n in [1,2,8]: abl.append(run("ablation_n_agents","qwen3_8b",u,t,i,n_agents=n))
for u in ab_units:
    for i in ABINT:
        for n in [1,2,8]: abl.append(run("ablation_chain_length","qwen3_8b",u,"sequential",i,n_agents=n,chain_length=n))
for u in ab_units:
    for i in ABINT:
        for pos in [1,3,4]: abl.append(run("ablation_target_position","qwen3_8b",u,"sequential",i,target_agent_position=pos))
for u in ab_units:
    for t in ["sequential","debate","scratchpad"]:
        for i in ABINT:
            for rho in [0.0,0.25,0.50]: abl.append(run("ablation_peer_evidence","qwen3_8b",u,t,i,peer_evidence=rho))
for u in ab_units:
    for i in ABINT:
        for r in [0,1,4]: abl.append(run("ablation_debate_rounds","qwen3_8b",u,"debate",i,debate_rounds=r))
for u in ab_units:
    for t in ["sequential","debate","scratchpad"]:
        for tau in [0.50,0.60,0.80,0.90]: abl.append(run("ablation_comm_threshold","qwen3_8b",u,t,"target_context_comm",comm_threshold=tau))

PARAM_METHODS=["gradient_ascent","lora_grad_diff"]
param=[]
train=[]
tid=0
for m in PARAM_MODELS:
    for u in ab_units:
        for method in PARAM_METHODS:
            tid += 1
            train.append({"training_id":f"TR{tid:04d}","model_id":m,"benchmark":u["benchmark"],"unit_id":u["unit_id"],"target_id":u["target_id"],"target_name":u["target_name"],"parameter_method":method,"status":"pending"})
            for t in TOPO:
                for comm in [False,True]:
                    param.append(run("parameter_unlearning",m,u,t,"parameter_plus_comm" if comm else "parameter_only",parameter_method=method))

rob=[]
for u in ab_units:
    for t in TOPO:
        for i in ABINT:
            for temp in [0.3,0.7]:
                for seed in [42,43,44,45,46]: rob.append(run("stochastic_robustness","qwen3_8b",u,t,i,temperature=temp,seed=seed))

wmdp=[run("wmdp_appendix",m,u,t,i) for m in ["qwen3_8b","mistral_7b"] for u in appendix_units for t in ["independent","debate","scratchpad"] for i in KEY]

save("runs_main.csv",runs_main)
save("runs_large_scale.csv",runs_large)
save("runs_coder_ablation.csv",runs_coder)
save("runs_ablations.csv",abl)
save("runs_parameter_unlearning.csv",param)
save("training_parameter_unlearning.csv",train)
save("runs_robustness.csv",rob)
save("runs_wmdp_appendix.csv",wmdp)
all_runs=runs_main+runs_large+runs_coder+abl+param+rob+wmdp
save("runs_all.csv",all_runs)
print("INFERENCE",len(all_runs),"TRAINING",len(train),"TOTAL",len(all_runs)+len(train))
