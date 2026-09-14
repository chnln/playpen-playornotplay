from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F


@contextmanager
def preserve_training_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def _resolve_token_id(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _gather_completion_logprobs(
    model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> list[float]:
    completion_length = full_ids.shape[1] - prompt_length
    if completion_length <= 0:
        return []

    logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, prompt_length - 1 : -1, :]
    shift_labels = full_ids[:, prompt_length:]
    token_logprobs = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logprobs[0].detach().cpu().tolist()


def generate_completion_with_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    device = model_device(model)
    encoded_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded_prompt["input_ids"].to(device)
    attention_mask = encoded_prompt["attention_mask"].to(device)
    prompt_length = input_ids.shape[1]

    generation_config = getattr(model, "generation_config", None)
    eos_token_id = _resolve_token_id(
        getattr(generation_config, "eos_token_id", None),
        tokenizer.eos_token_id,
    )
    pad_token_id = _resolve_token_id(
        getattr(generation_config, "pad_token_id", None),
        tokenizer.pad_token_id,
        eos_token_id,
    )

    do_sample = temperature > 0.0
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if do_sample:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    with preserve_training_mode(model), torch.no_grad():
        generated = model.generate(**generate_kwargs)
        completion_ids_tensor = generated[:, prompt_length:]
        full_attention_mask = torch.ones_like(generated, device=device)
        logprobs = _gather_completion_logprobs(
            model=model,
            full_ids=generated,
            attention_mask=full_attention_mask,
            prompt_length=prompt_length,
        )

    completion_ids = completion_ids_tensor[0].detach().cpu().tolist()
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": text,
    }
