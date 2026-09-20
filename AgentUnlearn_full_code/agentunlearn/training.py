from __future__ import annotations

import json
from pathlib import Path
from contextlib import contextmanager, nullcontext
from typing import Dict, Iterable, List, Tuple

import torch

from agentunlearn.config import ExperimentConfig
from agentunlearn.datasets import get_adapter
from agentunlearn.io import atomic_write_json, stable_sample
from agentunlearn.models import LocalChatModel
from agentunlearn.schemas import BenchmarkBundle, Probe


def _layers_container(model):
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.model.decoder.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None:
                return layers
        except Exception:
            pass
    return None


def configure_gradient_ascent_trainable(model, scope: str, last_n: int) -> List[str]:
    for p in model.parameters():
        p.requires_grad = False

    if scope == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif scope == "lm_head":
        for p in model.lm_head.parameters():
            p.requires_grad = True
    elif scope == "last_n_layers":
        layers = _layers_container(model)
        if layers is None:
            raise RuntimeError("Cannot locate transformer layers for last_n_layers gradient-ascent scope")
        for layer in list(layers)[-last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(model, "lm_head"):
            for p in model.lm_head.parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown gradient_ascent_scope={scope}")

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise RuntimeError("No trainable parameters selected")
    return names


def qa_loss(runtime: LocalChatModel, probe: Probe, max_length: int) -> torch.Tensor:
    model = runtime.model
    tok = runtime.tokenizer
    assert model is not None and tok is not None
    answer = probe.answers[0]
    prompt_messages = [
        {"role": "system", "content": "Answer the question accurately and concisely."},
        {"role": "user", "content": probe.prompt},
    ]
    prompt_text = runtime._render(prompt_messages, add_generation_prompt=True)
    full_text = runtime._render(prompt_messages + [{"role": "assistant", "content": answer}], add_generation_prompt=False)

    prompt_ids = tok(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
    full = tok(full_text, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = full["input_ids"].to(runtime.device)
    attention = full.get("attention_mask")
    if attention is not None:
        attention = attention.to(runtime.device)
    labels = input_ids.clone()
    prompt_len = min(prompt_ids.shape[1], input_ids.shape[1])
    labels[:, :prompt_len] = -100
    if (labels != -100).sum() == 0:
        raise RuntimeError("Training answer was entirely truncated; increase training.max_length")
    out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
    return out.loss


def _training_probes(bundle: BenchmarkBundle, cfg: ExperimentConfig) -> tuple[List[Probe], List[Probe]]:
    forget = stable_sample(
        bundle.direct_probes + bundle.indirect_probes,
        cfg.training.train_forget_examples,
        bundle.unit_id,
        "train-forget",
    )
    retain = stable_sample(
        bundle.retain_probes,
        cfg.training.train_retain_examples,
        bundle.unit_id,
        "train-retain",
    )
    if not forget:
        raise RuntimeError(f"No forget probes available for training {bundle.unit_id}")
    return forget, retain


def _atomic_torch_save(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        obj,
        tmp,
    )

    tmp.replace(path)


def _optimizer_to_device(
    optimizer,
    device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _save_training_resume(
    model,
    optimizer,
    method: str,
    output_dir: Path,
    next_step: int,
    losses: List[Dict],
    trainable_names: List[str] | None = None,
) -> None:

    state = {
        "method": method,
        "next_step": int(next_step),
        "loss_history": losses,
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    if method == "gradient_ascent":

        names = set(
            trainable_names or []
        )

        state["trainable_state"] = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if n in names
        }

    elif method == "lora_grad_diff":

        from peft import (
            get_peft_model_state_dict,
        )

        peft_state = (
            get_peft_model_state_dict(
                model
            )
        )

        state["peft_state"] = {
            k: v.detach().cpu()
            for k, v in peft_state.items()
        }

    else:
        raise ValueError(
            f"Unknown training method: {method}"
        )

    _atomic_torch_save(
        state,
        output_dir / "resume.pt",
    )


def _load_torch_checkpoint(
    path: Path,
):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def _restore_training_resume(
    model,
    optimizer,
    method: str,
    output_dir: Path,
) -> tuple[int, List[Dict]]:

    path = output_dir / "resume.pt"

    if not path.exists():
        return 0, []

    state = _load_torch_checkpoint(
        path
    )

    if state.get("method") != method:
        raise RuntimeError(
            "Training resume method mismatch: "
            f"{state.get('method')} != {method}"
        )

    if method == "gradient_ascent":

        named = dict(
            model.named_parameters()
        )

        for name, value in (
            state
            .get("trainable_state", {})
            .items()
        ):
            if name not in named:
                raise RuntimeError(
                    "Resume parameter missing "
                    f"from model: {name}"
                )

            p = named[name]

            with torch.no_grad():
                p.copy_(
                    value.to(
                        device=p.device,
                        dtype=p.dtype,
                    )
                )

    elif method == "lora_grad_diff":

        from peft import (
            set_peft_model_state_dict,
        )

        set_peft_model_state_dict(
            model,
            state.get(
                "peft_state",
                {},
            ),
        )

    optimizer.load_state_dict(
        state["optimizer_state"]
    )

    device = next(
        model.parameters()
    ).device

    _optimizer_to_device(
        optimizer,
        device,
    )

    rng = state.get(
        "torch_rng_state"
    )

    if rng is not None:
        torch.set_rng_state(rng)

    cuda_rng = state.get(
        "cuda_rng_state_all"
    )

    if (
        cuda_rng is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_rng
        )

    return (
        int(
            state.get(
                "next_step",
                0,
            )
        ),
        list(
            state.get(
                "loss_history",
                [],
            )
        ),
    )


def train_parameter_method(
    cfg: ExperimentConfig,
    model_id: str,
    model_path: str,
    benchmark: str,
    unit_id: str,
    target_id: str,
    target_name: str,
    method: str,
    output_dir: str | Path,
) -> Dict:

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter = get_adapter(
        benchmark,
        cfg.paths.resolve(
            cfg.paths.data_dir
        ),
    )

    bundle = adapter.load_bundle(
        unit_id,
        target_id,
        target_name,
    )

    forget, retain = _training_probes(
        bundle,
        cfg,
    )

    runtime = LocalChatModel(
        model_path,
        model_id,
        disable_qwen_thinking=(
            cfg.generation.disable_qwen_thinking
        ),
    ).load()

    model = runtime.model

    assert model is not None

    model.train()

    if hasattr(
        model.config,
        "use_cache",
    ):
        model.config.use_cache = False

    metadata = {
        "model_id": model_id,
        "model_path": str(model_path),
        "benchmark": benchmark,
        "unit_id": unit_id,
        "target_id": target_id,
        "target_name": target_name,
        "method": method,
        "max_steps":
            cfg.training.max_steps,
        "lr":
            cfg.training.lr,
        "retain_weight":
            cfg.training.retain_weight,
        "checkpoint_every":
            cfg.training.checkpoint_every,
    }

    trainable_names = None
    before = None

    if method == "gradient_ascent":

        trainable_names = (
            configure_gradient_ascent_trainable(
                model,
                cfg.training.gradient_ascent_scope,
                cfg.training.gradient_ascent_last_n_layers,
            )
        )

        metadata.update(
            gradient_ascent_scope=(
                cfg.training.gradient_ascent_scope
            ),
            gradient_ascent_last_n_layers=(
                cfg.training.gradient_ascent_last_n_layers
            ),
            trainable_parameter_names=(
                trainable_names
            ),
        )

        # This is the untouched base state.
        # It remains the reference even after resuming.
        before = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        optimizer = torch.optim.SGD(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            lr=cfg.training.lr,
        )

    elif method == "lora_grad_diff":

        try:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
            )
        except ImportError as e:
            raise RuntimeError(
                "lora_grad_diff requires "
                "the peft package"
            ) from e

        lora_cfg = LoraConfig(
            r=cfg.training.lora_r,
            lora_alpha=(
                cfg.training.lora_alpha
            ),
            lora_dropout=(
                cfg.training.lora_dropout
            ),
            bias="none",
            task_type=(
                TaskType.CAUSAL_LM
            ),
            target_modules=(
                cfg.training
                .lora_target_modules
            ),
        )

        runtime.model = (
            get_peft_model(
                model,
                lora_cfg,
            )
        )

        model = runtime.model
        model.train()

        optimizer = torch.optim.AdamW(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            lr=cfg.training.lr,
        )

        metadata.update(
            lora_r=
                cfg.training.lora_r,
            lora_alpha=
                cfg.training.lora_alpha,
            lora_dropout=
                cfg.training.lora_dropout,
            lora_target_modules=
                cfg.training
                .lora_target_modules,
        )

    else:
        runtime.unload()

        raise ValueError(
            "Unsupported parameter method: "
            f"{method}"
        )

    # --------------------------------------------------------------
    # Resume, if this training job was interrupted previously.
    # --------------------------------------------------------------

    start_step, losses = (
        _restore_training_resume(
            model,
            optimizer,
            method,
            output_dir,
        )
    )

    metadata[
        "resumed_from_step"
    ] = start_step

    if start_step > 0:
        print(
            f"  RESUME {method} "
            f"from step {start_step}/"
            f"{cfg.training.max_steps}",
            flush=True,
        )

    checkpoint_every = max(
        1,
        int(
            cfg.training.checkpoint_every
        ),
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    for step in range(
        start_step,
        cfg.training.max_steps,
    ):

        fp = forget[
            step % len(forget)
        ]

        forget_loss = qa_loss(
            runtime,
            fp,
            cfg.training.max_length,
        )

        if method == "gradient_ascent":

            objective = -forget_loss

            retain_loss_val = (
                float("nan")
            )

        else:

            if retain:

                rp = retain[
                    step % len(retain)
                ]

                retain_loss = qa_loss(
                    runtime,
                    rp,
                    cfg.training.max_length,
                )

                objective = (
                    -forget_loss
                    + cfg.training.retain_weight
                    * retain_loss
                )

                retain_loss_val = float(
                    retain_loss
                    .detach()
                    .cpu()
                )

            else:

                objective = -forget_loss

                retain_loss_val = (
                    float("nan")
                )

        optimizer.zero_grad(
            set_to_none=True
        )

        objective.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            cfg.training.gradient_clip,
        )

        optimizer.step()

        losses.append({
            "step": step,
            "forget_loss": float(
                forget_loss
                .detach()
                .cpu()
            ),
            "retain_loss":
                retain_loss_val,
            "objective": float(
                objective
                .detach()
                .cpu()
            ),
        })

        next_step = step + 1

        if (
            next_step % checkpoint_every
            == 0
            or next_step
            == cfg.training.max_steps
        ):

            _save_training_resume(
                model=model,
                optimizer=optimizer,
                method=method,
                output_dir=output_dir,
                next_step=next_step,
                losses=losses,
                trainable_names=(
                    trainable_names
                ),
            )

            print(
                f"    checkpoint "
                f"{next_step}/"
                f"{cfg.training.max_steps}",
                flush=True,
            )

    # --------------------------------------------------------------
    # Final parameter checkpoint
    # --------------------------------------------------------------

    if method == "gradient_ascent":

        assert before is not None

        delta = {}

        for n, p in (
            model.named_parameters()
        ):
            if n in before:

                delta[n] = (
                    p.detach().cpu()
                    - before[n]
                ).to(
                    torch.bfloat16
                )

        _atomic_torch_save(
            delta,
            output_dir / "delta.pt",
        )

    else:

        model.save_pretrained(
            output_dir / "adapter"
        )

    metadata[
        "completed_steps"
    ] = cfg.training.max_steps

    metadata[
        "loss_history"
    ] = losses

    atomic_write_json(
        output_dir / "metadata.json",
        metadata,
    )

    runtime.unload()

    return metadata


class DeltaParameterController:
    """Temporarily applies a saved parameter delta only for target-agent calls."""

    def __init__(self, runtime: LocalChatModel, delta_path: str | Path):
        if runtime.model is None:
            raise RuntimeError("Base model must be loaded before installing delta controller")
        raw = torch.load(delta_path, map_location="cpu")
        named = dict(runtime.model.named_parameters())
        self.items = []
        missing = []
        for n, d in raw.items():
            if n not in named:
                missing.append(n)
                continue
            p = named[n]
            # Keep delta on GPU to avoid host-to-device transfer on every target-agent call.
            self.items.append((p, d.to(device=p.device, dtype=p.dtype)))
        if missing:
            raise RuntimeError(f"Delta checkpoint parameters not found in base model: {missing[:5]}")
        self.active = False

    @contextmanager
    def context(self, is_target: bool):
        if not is_target:
            yield
            return
        if self.active:
            raise RuntimeError("Nested target parameter context is not supported")
        self.active = True
        with torch.no_grad():
            for p, d in self.items:
                p.add_(d)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, d in self.items:
                    p.sub_(d)
            self.active = False


class PeftParameterController:
    """Keeps LoRA enabled for target calls and disables it for peer calls."""

    def __init__(self, peft_model):
        self.model = peft_model

    def context(self, is_target: bool):
        if is_target:
            return nullcontext()
        return self.model.disable_adapter()


def apply_parameter_checkpoint(runtime: LocalChatModel, checkpoint_dir: str | Path, method: str) -> None:
    """Install a target-agent-only parameter intervention on a loaded runtime."""
    checkpoint_dir = Path(checkpoint_dir)
    if runtime.model is None:
        raise RuntimeError("Load base model before applying a parameter checkpoint")
    if method == "gradient_ascent":
        runtime.set_parameter_controller(DeltaParameterController(runtime, checkpoint_dir / "delta.pt"))
    elif method == "lora_grad_diff":
        try:
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError("Loading LoRA checkpoints requires peft") from e
        runtime.model = PeftModel.from_pretrained(runtime.model, checkpoint_dir / "adapter", is_trainable=False)
        runtime.model.eval()
        runtime.set_parameter_controller(PeftParameterController(runtime.model))
    else:
        raise ValueError(f"Unknown checkpoint method: {method}")
