#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


QWEN35_SIZES = {
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
}


def build_qwen_entry(model_name: str, huggingface_id: str) -> dict:
    return {
        "model_name": model_name,
        "backend": "huggingface_local",
        "huggingface_id": huggingface_id,
        "context_size": "262144",
        "model_config": {
            "premade_chat_template": True,
            "eos_to_cull": "<\\|im_end\\|>",
            "padding_side": "left",
            "trust_remote_code": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def build_registry() -> list[dict]:
    entries = [
        {
            "model_name": "smol-135m",
            "backend": "huggingface_local",
            "huggingface_id": "HuggingFaceTB/SmolLM-135M-Instruct",
            "context_size": "2048",
            "model_config": {
                "premade_chat_template": True,
                "eos_to_cull": "<\\|im_end\\|>",
                "padding_side": "left",
            },
        },
    ]
    for name, hf_id in QWEN35_SIZES.items():
        entries.append(build_qwen_entry(name, hf_id))
    return entries


def build_qwen_peft_entry(model_name: str, adapter_path: str, base_model: str = "Qwen/Qwen3.5-2B") -> dict:
    return {
        "model_name": model_name,
        "backend": "huggingface_local",
        "huggingface_id": base_model,
        "context_size": "262144",
        "model_config": {
            "premade_chat_template": True,
            "eos_to_cull": "<\\|im_end\\|>",
            "padding_side": "left",
            "trust_remote_code": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "peft_model": adapter_path,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="environment/playpen/model_registry.json",
        help="Path to the model_registry.json file to write.",
    )
    parser.add_argument("--qwen-peft-model-name")
    parser.add_argument("--qwen-peft-adapter-path")
    parser.add_argument("--qwen-base-model", default="Qwen/Qwen3.5-2B",
                        help="HuggingFace ID for the PEFT base model")
    parser.add_argument("--qwen-vanilla-model-name")
    parser.add_argument("--qwen-vanilla-hf-id",
                        help="HF repo id or local path evaluated as a plain (non-PEFT) Qwen model")
    args = parser.parse_args()

    registry = build_registry()
    if args.qwen_peft_model_name or args.qwen_peft_adapter_path:
        if not (args.qwen_peft_model_name and args.qwen_peft_adapter_path):
            raise SystemExit("--qwen-peft-model-name and --qwen-peft-adapter-path must be provided together")
        registry.append(build_qwen_peft_entry(
            args.qwen_peft_model_name, args.qwen_peft_adapter_path, args.qwen_base_model))
    if args.qwen_vanilla_model_name or args.qwen_vanilla_hf_id:
        if not (args.qwen_vanilla_model_name and args.qwen_vanilla_hf_id):
            raise SystemExit("--qwen-vanilla-model-name and --qwen-vanilla-hf-id must be provided together")
        registry.append(build_qwen_entry(args.qwen_vanilla_model_name, args.qwen_vanilla_hf_id))

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
