#!/usr/bin/env python
"""Build per-module-type ablated adapters for statscore diagnostic.

For each of the N distinct LoRA module types (q_proj, k_proj, v_proj, ...),
creates a variant where ONLY that module type's lora_B tensors are zeroed out,
while all others remain unchanged. This reveals which module types' deltas
are responsible for the static-benchmark regression.

Usage: ablate_lora_modules.py SRC_ADAPTER_DIR DST_BASE_DIR
  Creates DST_BASE_DIR/ablate-<module_type>/ for each module type found.
"""
import sys
import shutil
import json
import re
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "adapter_model.safetensors"

def extract_module_type(key):
    """Extract the module type from a PEFT key like
    base_model.model.model.layers.0.linear_attn.q_proj.lora_B.weight
    → 'q_proj' (the part immediately before 'lora_B')."""
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "lora_B" and i > 0:
            return parts[i - 1]
    return None


def main(src, dst_base):
    src, dst_base = Path(src), Path(dst_base)
    if not (src / WEIGHTS).exists():
        sys.exit(f"no {WEIGHTS} in {src}")

    cfg_path = src / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("use_dora"):
            sys.exit("DoRA adapter — ablation not supported")

    sd, meta = {}, {}
    with safe_open(str(src / WEIGHTS), framework="pt") as fh:
        meta = fh.metadata() or {}
        for k in fh.keys():
            sd[k] = fh.get_tensor(k)

    b_keys = [k for k in sd if "lora_B" in k]
    type_to_keys = defaultdict(list)
    for k in b_keys:
        mt = extract_module_type(k)
        if mt:
            type_to_keys[mt].append(k)
        else:
            print(f"WARNING: could not classify {k}")

    print(f"Found {len(b_keys)} lora_B tensors across {len(type_to_keys)} module types:")
    for mt, keys in sorted(type_to_keys.items()):
        norm = sum(sd[k].float().norm().item() ** 2 for k in keys) ** 0.5
        print(f"  {mt:20s}: {len(keys):3d} tensors, L2 norm = {norm:.4f}")

    non_b_files = [f for f in src.iterdir() if f.is_file() and f.name != WEIGHTS]

    for mt, keys in sorted(type_to_keys.items()):
        out = dst_base / f"ablate-{mt}"
        if out.exists():
            print(f"SKIP {out} (already exists)")
            continue
        out.mkdir(parents=True)
        for f in non_b_files:
            shutil.copy2(f, out / f.name)

        ablated = {}
        for k, v in sd.items():
            if k in keys:
                ablated[k] = torch.zeros_like(v)
            else:
                ablated[k] = v

        save_file(ablated, str(out / WEIGHTS), metadata=meta or {"format": "pt"})
        print(f"OK ablate-{mt}: zeroed {len(keys)} lora_B tensors -> {out}")

    print(f"\nDone. Run static eval on each ablate-* adapter to find which module type's delta hurts statscore most.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
