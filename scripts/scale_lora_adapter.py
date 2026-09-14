#!/usr/bin/env python
"""Scale a LoRA adapter's effective delta by a constant factor s.

Training-free LoRA delta scaling (Phase D): build interpolated
variants W_eff = W_base + s * Delta_lora without any training. Since the LoRA delta
is Delta = (alpha/r) * (lora_B @ lora_A), multiplying every lora_B tensor by s scales
the whole delta by s. adapter_config.json (incl. lora_alpha, r, target_modules) is
copied unchanged so the result loads as an ordinary PEFT adapter.

  s = 1.0 -> identical to the source adapter
  s = 0.0 -> base model behaviour

Usage: scale_lora_adapter.py SRC_ADAPTER_DIR DST_ADAPTER_DIR SCALE
"""
import sys
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "adapter_model.safetensors"


def main(src, dst, s):
    src, dst, s = Path(src), Path(dst), float(s)
    if dst.exists():
        sys.exit(f"refuse to overwrite existing {dst}")
    if not (src / WEIGHTS).exists():
        sys.exit(f"no {WEIGHTS} in {src} (only safetensors adapters supported)")
    dst.mkdir(parents=True)

    # explicit DoRA/QALoRA guard: pure lora_B scaling is only a clean
    # delta scale for plain LoRA. DoRA carries magnitude vectors; QALoRA differs too.
    import json
    cfg_path = src / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("use_dora"):
            sys.exit("adapter uses DoRA (use_dora=true): lora_B scaling is invalid; aborting")
        if cfg.get("use_qalora"):
            sys.exit("adapter uses QALoRA (use_qalora=true): lora_B scaling unsupported; aborting")

    # copy everything except the weights: adapter_config.json, tokenizer, chat template, ...
    for f in src.iterdir():
        if f.is_file() and f.name != WEIGHTS:
            shutil.copy2(f, dst / f.name)

    # load weights + preserve safetensors metadata (PEFT expects {"format": "pt"})
    sd, meta = {}, {}
    with safe_open(str(src / WEIGHTS), framework="pt") as fh:
        meta = fh.metadata() or {}
        for k in fh.keys():
            sd[k] = fh.get_tensor(k)

    # guard: only LoRA A/B deltas are expected. Bias deltas / modules_to_save would
    # NOT be scaled by this routine, silently changing behaviour -> abort instead.
    suspicious = [
        k for k in sd
        if ("modules_to_save" in k) or k.endswith(".bias")
        or ("lora_A" not in k and "lora_B" not in k)
    ]
    if suspicious:
        sys.exit(f"unexpected non-LoRA tensors, aborting: {suspicious[:8]}")

    n_b = 0
    for k in list(sd):
        if "lora_B" in k:
            t = sd[k]
            sd[k] = (t.float() * s).to(t.dtype)
            n_b += 1
    if n_b == 0:
        sys.exit("no lora_B tensors found - nothing scaled")

    save_file(sd, str(dst / WEIGHTS), metadata=meta or {"format": "pt"})
    print(f"OK scaled {n_b} lora_B tensors by s={s} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
