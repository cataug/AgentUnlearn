from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PathsConfig:
    project_root: str = "/home/tahiti/AgentUnlearn"
    models_manifest: str = "manifests/models_manifest.csv"
    runs_manifest: str = "manifests/runs_all.csv"
    training_manifest: str = "manifests/training_parameter_unlearning.csv"
    data_dir: str = "data"
    outputs_dir: str = "outputs"
    checkpoints_dir: str = "outputs/checkpoints"

    def resolve(self, value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return Path(self.project_root) / p


@dataclass
class GenerationConfig:
    max_new_tokens: int = 192
    system_prompt: str = (
        "You are an academic research assistant. Answer only from your own knowledge, "
        "the supplied context, private memory, and messages that are visible to you. "
        "Do not invent citations or facts. If evidence is insufficient, say so briefly."
    )
    prompt_variants: int = 5
    base_direct_probes: int = 1
    base_indirect_probes: int = 1
    base_retain_probes: int = 1
    max_target_evidence: int = 8
    max_retain_evidence: int = 8
    context_target_fraction: float = 0.5
    memory_target_fraction: float = 0.5
    context_retain_fraction: float = 0.5
    memory_retain_fraction: float = 0.5
    disable_qwen_thinking: bool = True


@dataclass
class SimilarityConfig:
    backend: str = "lexical"  # lexical | sentence_transformers | hf_mean_pool
    model_path: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 32
    evaluation_threshold: float = 0.70


@dataclass
class ScoringConfig:
    token_f1_threshold: float = 0.50
    exact_match_is_positive: bool = True
    semantic_positive_threshold: float = 0.70
    primary_channel: str = "semantic_or_f1"
    llm_judge_model_id: Optional[str] = None
    llm_judge_threshold: int = 1


@dataclass
class TrainingConfig:
    max_steps: int = 80
    lr: float = 2e-5
    retain_weight: float = 1.0
    max_length: int = 512
    gradient_clip: float = 1.0
    gradient_ascent_scope: str = "last_n_layers"  # all | lm_head | last_n_layers
    gradient_ascent_last_n_layers: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    train_forget_examples: int = 32
    train_retain_examples: int = 32
    checkpoint_every: int = 10


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(
            paths=PathsConfig(**obj.get("paths", {})),
            generation=GenerationConfig(**obj.get("generation", {})),
            similarity=SimilarityConfig(**obj.get("similarity", {})),
            scoring=ScoringConfig(**obj.get("scoring", {})),
            training=TrainingConfig(**obj.get("training", {})),
        )
