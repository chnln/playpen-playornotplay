#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model — in fp32, which is NOT optional.

Why fp32 is required here:
  The LoRA delta of a small-LR fine-tune (~1e-4 per element) sits below bf16's
  weight resolution (~2.4e-4 at magnitude ~0.05). Materializing W + s*BA in
  bf16 quantizes away part of the delta that adapter (bypass) serving preserves.
  Measured on our submission model: statscore 45.36 (adapter) -> 43.15 (bf16
  merge) -> 45.06 (fp32 merge), with the damage concentrated in ifeval (-10.9).

Two pitfalls this script avoids:
  1. `AutoPeftModelForCausalLM.from_pretrained(..., torch_dtype=...)` and
     `dtype=` are silently ignored by some transformers/peft versions — the
     base still loads in bf16. We upcast explicitly and assert.
  2. Saving without fixing config dtype would let `torch_dtype="auto"` loaders
     downcast on load. We record float32 in the config.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model HF id or path")
    ap.add_argument("--base-revision", help="Optional pinned Hugging Face base-model revision")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir")
    ap.add_argument("--out", required=True, help="Output dir for merged fp32 model")
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base, revision=args.base_revision).float()
    assert next(base.parameters()).dtype == torch.float32, "fp32 upcast failed"

    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    assert next(merged.parameters()).dtype == torch.float32, "merge left fp32"

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    # Match the evaluated 828e356 generation packaging, not only the weights.
    chat_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if chat_end is None or chat_end == tokenizer.unk_token_id:
        raise ValueError("The Qwen chat end-of-turn token is missing from the adapter tokenizer")
    old_eos = merged.generation_config.eos_token_id
    eos_ids = list(old_eos) if isinstance(old_eos, (list, tuple)) else ([] if old_eos is None else [old_eos])
    merged.generation_config.eos_token_id = list(dict.fromkeys(eos_ids + [chat_end]))
    merged.config.torch_dtype = torch.float32
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"fp32 merged model saved to {args.out}")
    print("Serve with torch_dtype='auto' (default). Do NOT downcast to bf16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
