---
base_model: Qwen/Qwen3.5-2B
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
language:
  - en
datasets:
  - colab-potsdam/playpen-data
tags:
  - lm-playschool
  - playpen
  - clembench
  - dialogue-games
model-index:
  - name: Qwen3.5-2B-playpen-playornotplay
    results:
      - task:
          type: text-generation
          name: Interactive dialogue games (clembench)
        dataset:
          name: Playpen public interactive suite (official final evaluation)
          type: colab-potsdam/playpen-data
          config: instances
          split: validation
        metrics:
          - type: clemscore
            name: clemscore
            value: 38.92
        source:
          name: LM Playschool 2026 final results (snapshot 2dd5a533)
          url: https://github.com/lm-playpen/lm-playschool-2026-final-results/blob/2dd5a53301ddca7fb3854af0847ca22962f5634d/summaries/summary_report_absolute.csv
      - task:
          type: text-generation
          name: Static benchmarks
        dataset:
          name: Playpen public static suite (official final evaluation)
          type: colab-potsdam/playpen-data
          config: instances-static
          split: validation
        metrics:
          - type: statscore
            name: statscore
            value: 44.14
        source:
          name: LM Playschool 2026 final results (snapshot 2dd5a533)
          url: https://github.com/lm-playpen/lm-playschool-2026-final-results/blob/2dd5a53301ddca7fb3854af0847ca22962f5634d/summaries/summary_report_absolute.csv
---

# Qwen3.5-2B Playpen Agent — team `playornotplay`

Submission for the **LM Playschool Challenge / Playpen shared task** (EMNLP 2026 Workshop) by team `playornotplay`.

We fine-tuned [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) with LoRA for text-only interactive dialogue-game play. Source trajectories and instances come from the `train` split of [colab-potsdam/playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data); preference training uses synthetic turn-local pairs and model-generated branch pairs. The [Hugging Face repository](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay/tree/828e356bc2f4bb4b200b533a49cc3094e1033fa0) provides **full merged weights** and the [scaled LoRA adapter](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay/tree/828e356bc2f4bb4b200b533a49cc3094e1033fa0/lora-adapter) used to produce them.

**Paper:** [Acquire, Repair, Preserve: A Diagnosis-Guided Post-Training Recipe for Small-Model Dialogue Game Agents](https://huggingface.co/papers/2608.28458) (arXiv [2608.28458](https://arxiv.org/abs/2608.28458))

## Official final evaluation

Results are from the [official LM Playschool 2026 leaderboard](https://github.com/lm-playpen/lm-playschool-2026-final-results), using snapshot [`2dd5a533`](https://github.com/lm-playpen/lm-playschool-2026-final-results/tree/2dd5a53301ddca7fb3854af0847ca22962f5634d) and its [absolute-score table](https://github.com/lm-playpen/lm-playschool-2026-final-results/blob/2dd5a53301ddca7fb3854af0847ca22962f5634d/summaries/summary_report_absolute.csv).

| Metric | Official Qwen3.5-2B baseline | This model | Difference |
| --- | ---: | ---: | ---: |
| Public Clemscore | 10.67 | **38.92** | +28.25 |
| Public Statscore | 44.24 | **44.14** | −0.10 |
| Closed in-domain Clemscore | 13.41 | **41.17** | +27.76 |
| Closed out-of-domain Clemscore | 3.72 | **7.88** | +4.16 |

The submitted model improves public and closed in-domain Clemscore while approximately preserving aggregate Statscore. Out-of-domain Clemscore remains low (7.88), with the largest gains concentrated in held-out Wordle variants. Aggregate static preservation does not imply improvement on every static benchmark.

The official evaluation used clemcore 3.7.2, temperature 0, a 5,000-token generation limit, and `enable_thinking=False`. Its public suite contains 72 interactive and 430 static episodes; the closed suites contain 1,272 in-domain and 360 out-of-domain episodes. The organizer's metadata does not record the exact clembench game-tree revision.

## Submitted model

We submitted the **fp32-merged model** `chnln/Qwen3.5-2B-playpen-playornotplay` at Hugging Face revision [`828e356`](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay/tree/828e356bc2f4bb4b200b533a49cc3094e1033fa0).

## How to load

The submitted checkpoint stores the merged weights in fp32. Load it with `torch_dtype="auto"` to retain its stored precision.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "chnln/Qwen3.5-2B-playpen-playornotplay"
revision = "828e356bc2f4bb4b200b533a49cc3094e1033fa0"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype="auto",  # preserve the stored fp32 weights
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
```

Use the bundled chat template with `enable_thinking=False` when formatting game messages.

## Development results

We evaluated on the `validation` split of [colab-potsdam/playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data) at revision [`557d8caf`](https://huggingface.co/datasets/colab-potsdam/playpen-data/tree/557d8cafd1bc4557bc22803a6d4502ef53fb664c), using the `instances` configuration. The suite contains 67 interactive episodes across 14 games.

The evaluator used clembench tree [`ed39486`](https://github.com/clp-research/clembench/tree/ed3948604695b077d285654b73fcd8aae7a12bff). These results are not directly comparable with the official final evaluation, which used a different protocol.

| Phase or checkpoint | Method | Clemscore |
| --- | --- | ---: |
| Base | Qwen3.5-2B | 13.05 |
| A · Acquire | Broad SFT | 43.85 |
| B1 · Repair | Turn-DPO: repeat / length | 46.83 |
| B2 · Repair | Turn-DPO: invalid / duplicate | 45.52 |
| C · Refine | Branch-DPO | 48.52 |
| D · Preserve | Delta scaling, `s=0.85` | 50.43† |
| Submitted model | fp32 merge | 48.70 |

† Mean of two runs on the same host (50.82 and 50.03). All Clemscores cover the full 14-game development suite. Phases B and C train on Wordle data only.

## Training summary

The effective pipeline has four phases:

| Phase | Method | Data or operation |
| --- | --- | --- |
| A · Acquire | Broad SFT | Successful Playpen training episodes, capped at 700 per game |
| B · Repair | Two passes of turn-local DPO | Wordle: repeated guesses / bad length, then invalid words / duplicate answer fields |
| C · Refine | Branch-DPO | Immediate diverging responses from the current model's branched Wordle trajectories, paired by downstream episode score |
| D · Preserve | Training-free delta scaling and fp32 merge | Scale the learned LoRA delta by `s=0.85`, then merge into the base |

Earlier descriptions included two nominal stages—weak-game SFT and Wordle GRPO—that subsequent verification showed had applied no parameter updates. The four phases above describe the effective pipeline that produced the evaluated weights.

**Training configuration.** All trainable phases use LoRA rank 16, alpha 32, dropout 0.05, all-linear targets, AdamW, a cosine schedule with warmup ratio 0.03, one epoch, and bf16 training. Data shuffling/splitting uses seed 42; a 5% holdout is taken from the constructed training corpus, not from the challenge validation split.

| Phase | Learning rate | DPO beta | Micro-batch × accumulation | Max length | Gradient checkpointing |
| --- | ---: | ---: | --- | ---: | --- |
| A · Broad SFT | **2e-4** | — | 8 × 4 = 32 | 2,048 | On |
| B1 · Turn-DPO, repeat / length | 5e-6 | 0.1 | 1 × 16 = 16 | 1,024 | Off |
| B2 · Turn-DPO, invalid / duplicate | 5e-6 | 0.1 | 1 × 16 = 16 | 1,024 | Off |
| C · Branch-DPO | 5e-6 | 0.2 | 1 × 16 = 16 | 1,024 | On |

See the [paper](https://arxiv.org/abs/2608.28458v1) for data and preference construction, model selection and packaging, compute, and reproducibility details.

## Citation

If you use this model or training recipe, please cite our [paper](https://arxiv.org/abs/2608.28458v1):

```bibtex
@misc{li2026acquirerepairpreserve,
  title = {Acquire, Repair, Preserve: A Diagnosis-Guided Post-Training Recipe for Small-Model Dialogue Game Agents},
  author = {Nan Li},
  year = {2026},
  eprint = {2608.28458},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2608.28458v1}
}
```
