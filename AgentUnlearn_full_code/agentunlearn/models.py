from __future__ import annotations

import gc
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentunlearn.io import set_global_seed


class LocalChatModel:
    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        disable_qwen_thinking: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.model_path = str(Path(model_path).resolve())
        self.model_id = model_id
        self.disable_qwen_thinking = disable_qwen_thinking
        self.dtype = dtype
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_seconds = None
        self.parameter_controller = None

    def load(self) -> "LocalChatModel":
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(local_files_only=True, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                **kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                **kwargs,
            )
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.time() - t0
        return self

    def set_parameter_controller(self, controller) -> None:
        self.parameter_controller = controller

    def agent_parameter_context(self, is_target: bool):
        if self.parameter_controller is None:
            return nullcontext()
        return self.parameter_controller.context(is_target)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.parameter_controller = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _render(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        assert self.tokenizer is not None
        kwargs: Dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.disable_qwen_thinking:
            kwargs["enable_thinking"] = False
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        seed: int,
        max_new_tokens: int,
    ) -> tuple[str, Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        set_global_seed(seed)
        prompt = self._render(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
        else:
            generation_kwargs.update(do_sample=False)

        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **generation_kwargs)
        elapsed = time.time() - t0
        new_tokens = out[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        meta = {
            "generation_sec": elapsed,
            "input_tokens": int(input_len),
            "output_tokens": int(new_tokens.numel()),
            "temperature": float(temperature),
            "seed": int(seed),
        }
        return text, meta

    def answer_log_likelihood(
        self,
        messages_without_answer: List[Dict[str, str]],
        answer: str,
    ) -> float:
        """Average token log-likelihood of `answer` conditioned on a chat prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        prompt_text = self._render(messages_without_answer, add_generation_prompt=True)
        full_messages = list(messages_without_answer) + [{"role": "assistant", "content": answer}]
        full_text = self._render(full_messages, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        full = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = full["input_ids"].to(self.device)
        attention_mask = full.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        prompt_len = min(prompt_ids.shape[1], input_ids.shape[1])
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        if (labels != -100).sum() == 0:
            return float("nan")

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return float(-outputs.loss.detach().cpu().item())

    def metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "dtype": str(self.dtype),
            "device": self.device,
            "load_seconds": self.load_seconds,
        }
        if torch.cuda.is_available():
            meta.update(
                gpu_name=torch.cuda.get_device_name(0),
                gpu_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
                gpu_reserved_gib=torch.cuda.memory_reserved() / 1024**3,
            )
        return meta
