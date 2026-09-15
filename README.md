# playornotplay — Playpen shared-task code

We fine-tuned **Qwen3.5-2B** for interactive dialogue-game play in the LM
Playschool Challenge / Playpen shared task (EMNLP 2026 Workshop). Training
combines broad supervised fine-tuning, two Wordle turn-DPO passes and Wordle
branch-DPO, followed by LoRA-delta scaling and an fp32 merge.

This repository contains the training and evaluation code. The
[model card and weights](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay)
are hosted on Hugging Face, and the method is described in our
[paper](https://arxiv.org/abs/2608.28458v1).

- [Submitted model and results](#submitted-model-and-results)
- [Training recipe](#training-recipe)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository layout](#repository-layout)
- [License and third-party material](#license-and-third-party-material)
- [Citation](#citation)

## Submitted model and results

We submitted the **fp32-merged model** at Hugging Face revision
[`828e356`](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay/tree/828e356bc2f4bb4b200b533a49cc3094e1033fa0).
Load it with `torch_dtype="auto"` to retain its stored precision, and format
game messages with `enable_thinking=False`.

| Metric | Official Qwen3.5-2B base | Submitted model | Difference |
| --- | ---: | ---: | ---: |
| Public Clemscore | 10.67 | **38.92** | +28.25 |
| Public Statscore | 44.24 | **44.14** | −0.10 |
| Closed in-domain Clemscore | 13.41 | **41.17** | +27.76 |
| Closed out-of-domain Clemscore | 3.72 | **7.88** | +4.16 |

Source: [official leaderboard](https://github.com/lm-playpen/lm-playschool-2026-final-results),
[absolute scores at snapshot 2dd5a533](https://github.com/lm-playpen/lm-playschool-2026-final-results/blob/2dd5a53301ddca7fb3854af0847ca22962f5634d/summaries/summary_report_absolute.csv).
Public and closed in-domain gains accompany approximately preserved aggregate
static performance. Out-of-domain Clemscore remains low, with the largest gains
concentrated in held-out Wordle variants. The model card also reports the
[phase-by-phase development results](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay#development-results).

## Training recipe

| Phase | Operation | Code |
| --- | --- | --- |
| A · Acquire | Broad success-only SFT, at most 700 episodes per game | [SFT trainer](trainers/qwen35_2b_sft_lora.py) |
| B · Repair | Two Wordle turn-DPO passes: repeat/length, then invalid/duplicate | [Turn-DPO trainer](trainers/qwen35_2b_wordle_turn_dpo_lora.py) |
| C · Refine | Wordle branch-DPO on the model's own branched trajectories | [Branch-DPO trainer](trainers/qwen35_2b_wordle_branch_dpo_lora.py) |
| D · Preserve | Scale the LoRA delta by 0.85, then merge in fp32 | [Scale](scripts/scale_lora_adapter.py), [merge](scripts/merge_lora_fp32.py) |

These four phases produced the submitted checkpoint at Hugging Face revision
[`828e356`](https://huggingface.co/chnln/Qwen3.5-2B-playpen-playornotplay/tree/828e356bc2f4bb4b200b533a49cc3094e1033fa0).

All trainable phases use LoRA rank 16, alpha 32, dropout 0.05, all-linear
targets, AdamW, a cosine schedule with warmup ratio 0.03, one epoch, bf16
training and seed 42. [configs/recipe.json](configs/recipe.json) records every
setting used by the launcher.

| Phase | Learning rate | DPO beta | Micro-batch × accumulation | Max length | Gradient checkpointing |
| --- | ---: | ---: | --- | ---: | --- |
| A · Broad SFT | 2e-4 | — | 8 × 4 = 32 | 2,048 | On |
| B1 · Turn-DPO, repeat / length | 5e-6 | 0.1 | 1 × 16 = 16 | 1,024 | Off |
| B2 · Turn-DPO, invalid / duplicate | 5e-6 | 0.1 | 1 × 16 = 16 | 1,024 | Off |
| C · Branch-DPO | 5e-6 | 0.2 | 1 × 16 = 16 | 1,024 | On |

See the [paper](https://arxiv.org/abs/2608.28458v1) for data and preference
construction, model selection, compute and reproducibility details.

## Installation

Use a Linux machine with an NVIDIA GPU, SLURM, Python 3.12 and
[`uv`](https://github.com/astral-sh/uv). The default batch sizes target one
A100-80GB GPU. Run these commands from the repository root:

```bash
# Playpen, with a small patch that lets PLAYPEN_DATASET_REVISION pin evaluation data
git clone https://github.com/lm-playpen/playpen.git environment/playpen
git -C environment/playpen checkout b69702bf32ea50a1e74ec4ead288b242adfb0ae8
git -C environment/playpen apply ../../patches/playpen-dataset-revision.patch

# clembench games, with the static benchmarks from a later commit for clemcore compatibility
git clone https://github.com/clp-research/clembench.git environment/playpen/clembench
git -C environment/playpen/clembench checkout ed3948604695b077d285654b73fcd8aae7a12bff
git -C environment/playpen/clembench restore \
  --source b716d279ae70befc988bffff2fb1499ccc81866a --worktree -- static

# Wordle guess list used to build phase-B2 negatives (12,953 words)
curl -L -o environment/playpen/clembench/wordle/resources/target_words/en/official_recognized_words.txt \
  https://raw.githubusercontent.com/3b1b/videos/b1ad11dfb3d38a9f5d3f1c9e4548208be51e1b96/_2022/wordle/data/allowed_words.txt
echo "786dd7405eedf985eefada40d7d5ab21894785cfd5ca2c559687d3388cd6a527  environment/playpen/clembench/wordle/resources/target_words/en/official_recognized_words.txt" \
  | sha256sum -c -

# Python environment
uv venv --python 3.12 environment/playpen/venv_repro
uv pip install --python environment/playpen/venv_repro/bin/python \
  -r requirements-training.txt \
  -r environment/playpen/clembench/requirements.txt \
  -e 'environment/playpen[trl]'
```

Phase B2 uses the Wordle guess list
([`allowed_words.txt` from 3b1b/videos at commit `b1ad11d`](https://github.com/3b1b/videos/blob/b1ad11dfb3d38a9f5d3f1c9e4548208be51e1b96/_2022/wordle/data/allowed_words.txt))
to construct invalid-word negatives.

[requirements-training.txt](requirements-training.txt) pins the direct
dependencies of our training environment (torch 2.6.0, transformers 5.4.0,
PEFT 0.18.1, TRL 0.29.1, clemcore 3.7.2). Transitive dependencies and CUDA are
not locked; check driver compatibility on your machine.

Then set the environment and authenticate Weights & Biases, which the training
launcher uses by default:

```bash
export PLAYPEN_VENV_DIR="$PWD/environment/playpen/venv_repro"
export CLEMBENCH_HOME="$PWD/environment/playpen"
"$PLAYPEN_VENV_DIR/bin/playpen" list games
"$PLAYPEN_VENV_DIR/bin/wandb" login
```

A quick offline check, which loads no model and submits nothing:

```bash
"$PLAYPEN_VENV_DIR/bin/python" -m unittest discover -s tests
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py A --dry-run
```

Before full experiments, test the installation with a small SLURM training and
evaluation run. The SLURM partition defaults to `gpua100` and can be changed
with `PLAYPEN_SLURM_PARTITION`.

## Training

[`scripts/submit_recipe_phase.py`](scripts/submit_recipe_phase.py) submits one
phase with the settings from `configs/recipe.json`. It ignores other `PLAYPEN_*`
variables inherited from your shell and prints the job ID, output directory and
log path. Every training and evaluation job writes `run_metadata.json`. Use `--dry-run` to
inspect a phase and `--time HH:MM:SS` to override its wall time.

```bash
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py A
```

When a job completes, check its log, `trainer_state.json` and adapter files,
then pass its `final/` directory as the parent of the next phase:

```bash
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py B1 --parent /absolute/path/to/A/final
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py B2 --parent /absolute/path/to/B1/final
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py C --parent /absolute/path/to/B2/final
```

Training reads the `train` split of
[colab-potsdam/playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data),
pinned to revision `cf23af92` for phases A, B1 and B2 and `557d8caf` for
phase C.

Phase C saves its generated pairs to `artifacts/branch-<timestamp>/pairs.jsonl.gz`
before training on them. Pair generation samples at temperature 0.7, so pairs
vary between runs and retraining will not reproduce the released weights
exactly.

Phase D needs no GPU training. Scale the **unscaled** Phase-C adapter once, then
merge it into the base model in fp32 (use `--base-revision` to pin the base
download):

```bash
"$PLAYPEN_VENV_DIR/bin/python" scripts/scale_lora_adapter.py \
  /absolute/path/to/C/final artifacts/beta0p20-s085 0.85
"$PLAYPEN_VENV_DIR/bin/python" scripts/merge_lora_fp32.py \
  --base Qwen/Qwen3.5-2B --adapter artifacts/beta0p20-s085 --out artifacts/merged-fp32
```

## Evaluation

To evaluate the submitted model, download its revision (about 7.5 GB) and
register the local directory:

```bash
"$PLAYPEN_VENV_DIR/bin/hf" download chnln/Qwen3.5-2B-playpen-playornotplay \
  --revision 828e356bc2f4bb4b200b533a49cc3094e1033fa0 \
  --local-dir artifacts/evaluated-model
export PLAYPEN_QWEN_VANILLA_MODEL_NAME=playornotplay-evaluated
export PLAYPEN_QWEN_VANILLA_HF_ID="$PWD/artifacts/evaluated-model"
export PLAYPEN_DATASET_REVISION=557d8cafd1bc4557bc22803a6d4502ef53fb664c
unset PLAYPEN_QWEN_PEFT_MODEL_NAME PLAYPEN_QWEN_PEFT_ADAPTER_PATH
bash scripts/submit_playpen_eval.sh playornotplay-evaluated clem "" 02:30:00
bash scripts/submit_playpen_eval.sh playornotplay-evaluated static "" 02:30:00
```

For a trained adapter, set `PLAYPEN_QWEN_PEFT_MODEL_NAME` and
`PLAYPEN_QWEN_PEFT_ADAPTER_PATH` instead of the two `VANILLA` variables.
[scripts/summarize_clem_eval.py](scripts/summarize_clem_eval.py) summarizes the
results.

These commands run our development evaluation on the `validation` split of
[playpen-data at revision 557d8caf](https://huggingface.co/datasets/colab-potsdam/playpen-data/tree/557d8cafd1bc4557bc22803a6d4502ef53fb664c)
with temperature 0 and the Playpen default of 300 generated tokens
(`PLAYPEN_EVAL_TEMPERATURE` and `PLAYPEN_EVAL_MAX_TOKENS` override them). The
official evaluation used a 5,000-token limit, a public suite with 72
interactive episodes, and separate closed suites, so these scores are not
directly comparable with the official results.

## Repository layout

| Path | Contents |
| --- | --- |
| `trainers/` | SFT, turn-DPO and branch-DPO trainers with shared helpers |
| `scripts/` | Phase launcher, SLURM submitters, registry setup, packaging and result summaries |
| `scripts/jobs/` | SLURM job scripts for training and evaluation |
| `configs/` | Training recipe and submitted-model identity |
| `patches/` | Dataset-revision patch for the pinned Playpen CLI |
| `tests/` | Offline tests for the launcher, registry and fp32 packaging |

The sweep launchers (`scripts/submit_wordle_*_batches.sh`), the GRPO trainer
and `scripts/ablate_lora_modules.py` are exploratory tools and are not needed
for the recipe above.

## License and third-party material

Our code is released under [Apache-2.0](LICENSE). Upstream material keeps its
own terms and is not bundled:

- [Playpen](https://github.com/lm-playpen/playpen) (MIT): the trainers build on
  its interfaces, `trainers/data_utils.py` derives from its example helper, and
  the patch modifies its CLI; see the [MIT notice](LICENSES/playpen-MIT.txt).
- [clembench](https://github.com/clp-research/clembench) and clemcore: installed
  from upstream with their own licenses and game resources.
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) base weights and
  [playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data):
  downloaded from Hugging Face under their own terms.
- The [Wordle guess list](https://github.com/3b1b/videos/blob/b1ad11dfb3d38a9f5d3f1c9e4548208be51e1b96/_2022/wordle/data/allowed_words.txt)
  is downloaded during installation from
  [3b1b/videos](https://github.com/3b1b/videos), whose contents are licensed
  under CC BY-NC-SA 4.0; it is not redistributed here.

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
