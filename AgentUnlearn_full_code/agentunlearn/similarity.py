from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np

from agentunlearn.config import SimilarityConfig
from agentunlearn.io import tokenize_simple


class SimilarityEngine:
    def __init__(self, cfg: SimilarityConfig):
        self.cfg = cfg
        self.backend = cfg.backend
        self.model = None
        self.tokenizer = None
        self._reference_embedding_cache = {}

        if self.backend == "sentence_transformers":
            if not cfg.model_path:
                raise ValueError("sentence_transformers backend requires similarity.model_path")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(cfg.model_path, device=cfg.device)
        elif self.backend == "hf_mean_pool":
            if not cfg.model_path:
                raise ValueError("hf_mean_pool backend requires similarity.model_path")
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
            try:
                self.model = AutoModel.from_pretrained(
                    cfg.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                ).to(cfg.device).eval()
            except TypeError:
                self.model = AutoModel.from_pretrained(
                    cfg.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).to(cfg.device).eval()
        elif self.backend != "lexical":
            raise ValueError(f"Unknown similarity backend: {self.backend}")

    @staticmethod
    def _lexical_cosine(a: str, b: str) -> float:
        ca, cb = Counter(tokenize_simple(a)), Counter(tokenize_simple(b))
        if not ca or not cb:
            return 0.0
        dot = sum(v * cb.get(k, 0) for k, v in ca.items())
        na = math.sqrt(sum(v * v for v in ca.values()))
        nb = math.sqrt(sum(v * v for v in cb.values()))
        return float(dot / (na * nb)) if na and nb else 0.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "sentence_transformers":
            arr = self.model.encode(
                list(texts),
                batch_size=self.cfg.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(arr, dtype=np.float32)

        if self.backend == "hf_mean_pool":
            import torch
            chunks: List[np.ndarray] = []
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = list(texts[i:i + self.cfg.batch_size])
                x = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.cfg.device)
                with torch.inference_mode():
                    y = self.model(**x).last_hidden_state
                mask = x["attention_mask"].unsqueeze(-1)
                pooled = (y * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
                chunks.append(pooled.cpu().numpy())
            return np.concatenate(chunks, axis=0)

        raise RuntimeError("encode() is not used by lexical backend")

    def pairwise(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> np.ndarray:
        if self.backend == "lexical":
            out = np.zeros((len(texts_a), len(texts_b)), dtype=np.float32)
            for i, a in enumerate(texts_a):
                for j, b in enumerate(texts_b):
                    out[i, j] = self._lexical_cosine(a, b)
            return out
        a = self.encode(texts_a)
        b = self.encode(texts_b)
        return a @ b.T

    def max_similarity(self, text: str, references: Sequence[str]) -> float:
        refs = [r for r in references if r and r.strip()]
        if not text.strip() or not refs:
            return 0.0
        if self.backend == "lexical":
            return float(self.pairwise([text], refs).max())
        key = tuple(refs)
        ref_emb = self._reference_embedding_cache.get(key)
        if ref_emb is None:
            ref_emb = self.encode(refs)
            # Experiment units are few; a small bounded cache avoids repeatedly encoding target references.
            if len(self._reference_embedding_cache) >= 64:
                self._reference_embedding_cache.pop(next(iter(self._reference_embedding_cache)))
            self._reference_embedding_cache[key] = ref_emb
        text_emb = self.encode([text])
        return float((text_emb @ ref_emb.T).max())

    def filter_text(
        self,
        text: str,
        references: Sequence[str],
        threshold: float,
    ) -> tuple[str, List[dict]]:
        # Sentence-level filtering preserves unrelated content in a message.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        kept: List[str] = []
        removed: List[dict] = []
        for s in sentences:
            score = self.max_similarity(s, references)
            if score >= threshold:
                removed.append({"text": s, "similarity": score})
            else:
                kept.append(s)
        if not kept and removed:
            return "[Target-related content filtered]", removed
        return " ".join(kept), removed
