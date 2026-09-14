# playornotplay — Playpen shared-task code

We fine-tuned **Qwen3.5-2B** for interactive dialogue-game play in the LM
Playschool Challenge / Playpen shared task (EMNLP 2026 Workshop). Training
combines broad supervised fine-tuning, two Wordle turn-DPO passes, and Wordle
branch-DPO, followed by LoRA-delta scaling and an fp32 merge.

This repository provides the training and evaluation code, phase
configurations and documentation; the
[training summary](docs/MODEL_CARD.md#training-summary) lists the
hyperparameters. The [merged model and adapter](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay/tree/828e356bc2f4bb4b200b533a49cc3094e1033fa0)
are hosted on Hugging Face.

## Submitted model and official results

We submitted the **fp32-merged model** at Hugging Face revision
[`828e356`](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay/tree/828e356bc2f4bb4b200b533a49cc3094e1033fa0).
Load it with `torch_dtype="auto"` to retain its stored precision.

| Metric | Official Qwen3.5-2B base | Submitted model | Difference |
| --- | ---: | ---: | ---: |
| Public Clemscore | 10.67 | **38.92** | +28.25 |
| Public Statscore | 44.24 | **44.14** | −0.10 |
| Closed in-domain Clemscore | 13.41 | **41.17** | +27.76 |
| Closed out-of-domain Clemscore | 3.72 | **7.88** | +4.16 |

Source: [official leaderboard](https://github.com/lm-playpen/lm-playschool-2026-final-results),
[absolute scores at snapshot 2dd5a533](https://github.com/lm-playpen/lm-playschool-2026-final-results/blob/2dd5a53301ddca7fb3854af0847ca22962f5634d/summaries/summary_report_absolute.csv).
Public and closed in-domain gains accompany approximately preserved aggregate
static performance. Out-of-domain Clemscore remains low (7.88), with the largest
gains concentrated in held-out Wordle variants.

The [development trajectory](docs/MODEL_CARD.md#development-results) reports
the phase-by-phase Clemscores from the paper on the `validation` split of
[colab-potsdam/playpen-data at revision 557d8caf](https://huggingface.co/datasets/colab-potsdam/playpen-data/tree/557d8cafd1bc4557bc22803a6d4502ef53fb664c).
This development protocol differs from the official evaluation.

## Effective recipe

| Phase | Operation | Code |
| --- | --- | --- |
| A · Acquire | Broad success-only SFT, LR 2e-4 | [SFT trainer](trainers/qwen35_2b_sft_lora.py) |
| B · Repair | Two Wordle turn-DPO passes: repeat/length, then invalid/duplicate; beta 0.1 | [Turn-DPO trainer](trainers/qwen35_2b_wordle_turn_dpo_lora.py) |
| C · Refine | Wordle branch-DPO, beta 0.2, gradient checkpointing on | [Branch-DPO trainer](trainers/qwen35_2b_wordle_branch_dpo_lora.py) |
| D · Preserve | Scale LoRA delta by 0.85, then fp32 merge | [Scale](scripts/scale_lora_adapter.py), [merge](scripts/merge_lora_fp32.py) |

Earlier descriptions included two nominal stages—weak-game SFT and Wordle GRPO—that subsequent verification showed had applied no parameter updates. The four phases above describe the effective pipeline that produced the evaluated weights.

The [phase configuration](configs/recipe.json) records the training settings.
See our [paper](https://arxiv.org/abs/2608.28458v1) for data and preference
construction, model selection and packaging, compute, and reproducibility
details.

## Start here

1. Follow [setup and runtime provenance](docs/SETUP.md) to create an isolated
   environment and install pinned upstream sources plus the evaluation patch.
2. Read the [training and evaluation commands](docs/TRAINING_PROCESS.md).
3. Preview a phase without submitting anything:

```bash
environment/playpen/venv_repro/bin/python scripts/submit_recipe_phase.py A --dry-run
```

Remove `--dry-run` only when ready to submit to your SLURM scheduler. Run later
phases against the preceding phase's completed adapter, using `--parent`.
The canonical configurations target one A100-80GB; adjust your resource plan
before using a smaller GPU. Every train/eval wrapper writes `run_metadata.json`.

Additional GRPO and DPO sweep scripts cover exploratory experiments outside
the submitted pipeline. Use `submit_recipe_phase.py` for the four-phase
sequence. [Development comparisons](docs/negative_results.md) report
checkpoint-level results from separate experiments.

## Repository contents and license

The repository contains source code, configurations and documentation. Model
weights are hosted on Hugging Face; datasets and large run outputs are not
bundled. Runtime requirements and test commands are described in the
[setup guide](docs/SETUP.md).

Project additions are provided under [Apache-2.0](LICENSE). Upstream code and
resources retain their own terms; see [third-party notices](docs/THIRD_PARTY.md).

## Citation

If you use this model or code, please cite our [paper](https://arxiv.org/abs/2608.28458v1):

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
