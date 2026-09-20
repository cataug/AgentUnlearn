import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

path = Path(sys.argv[1]).resolve()
name = Path(sys.argv[1]).name

print("=" * 88, flush=True)
print("MODEL       :", name, flush=True)
print("PATH        :", path, flush=True)
print("torch       :", torch.__version__, flush=True)
print("transformers:", transformers.__version__, flush=True)
print("CUDA        :", torch.cuda.is_available(), flush=True)

if not torch.cuda.is_available():
    print("RESULT      : FAIL — CUDA unavailable", flush=True)
    sys.exit(1)

print("GPU         :", torch.cuda.get_device_name(0), flush=True)
free, total = torch.cuda.mem_get_info()
print(f"GPU free    : {free/1024**3:.2f} / {total/1024**3:.2f} GiB", flush=True)

try:
    t0 = time.time()

    print("\n[1/4] tokenizer", flush=True)
    tok = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
    )

    print("[2/4] model", flush=True)

    kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
    )

    # transformers 5.x uses dtype; older releases used torch_dtype.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            dtype=torch.bfloat16,
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

    model = model.to("cuda")
    model.eval()

    print(
        f"      allocated={torch.cuda.memory_allocated()/1024**3:.2f} GiB",
        flush=True,
    )

    print("[3/4] prompt", flush=True)

    messages = [{
        "role": "user",
        "content": "Reply with exactly: LOCAL MODEL SMOKE TEST PASSED"
    }]

    try:
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tok(prompt, return_tensors="pt").to("cuda")

    print("[4/4] generate", flush=True)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    generated = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(generated, skip_special_tokens=True).strip()

    print("\nOUTPUT:")
    print(text)

    print(f"\nTIME        : {time.time()-t0:.1f} s")
    print(
        f"PEAK GPU    : {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB"
    )
    print("RESULT      : PASS", flush=True)

except Exception as e:
    print("\nRESULT      : FAIL", flush=True)
    print(f"ERROR       : {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
