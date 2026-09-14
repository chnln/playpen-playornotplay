# Training and evaluation commands

After [setting up the environment](SETUP.md), run phases A, B1, B2 and C in
order, then scale and merge the adapter in phase D. See the
[training configuration](MODEL_CARD.md#training-summary) for hyperparameters
and the [paper](https://arxiv.org/abs/2608.28458v1) for methodology, data
construction, model selection and reproducibility.

## Train the effective sequence

`scripts/submit_recipe_phase.py` selects a phase from
[configs/recipe.json](../configs/recipe.json) and sets its training environment.
It removes inherited experimental `PLAYPEN_*` settings.
Only `PLAYPEN_VENV_DIR` and `PLAYPEN_SLURM_PARTITION` carry through as execution
settings; Hugging Face and W&B credentials are retained without being printed.

```bash
export PLAYPEN_VENV_DIR="$PWD/environment/playpen/venv_repro"
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py A --dry-run
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py A
```

The submission command prints its job ID, output directory and log path.
Wait for that job to complete and inspect its log, `trainer_state.json`, adapter
files and metadata before using its `final/` directory as the next parent.

For each continuation, replace the example absolute path with the output of
the preceding phase. Each command submits a GPU job and requires a completed
parent adapter.

```bash
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py B1 --parent /absolute/path/to/A/final
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py B2 --parent /absolute/path/to/B1/final
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py C --parent /absolute/path/to/B2/final
```

Use `--dry-run` with a real parent path to inspect a continuation without
submitting. `--time HH:MM:SS` explicitly overrides the default wall time.
The launcher defaults to W&B project `playpen-wordle` and a descriptive unique
run name; authenticate W&B before submission.

The configuration uses the `train` split of
[colab-potsdam/playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data),
pinning A, B1 and B2 to revision `cf23af92` and C to `557d8caf`.
The trainers read `PLAYPEN_TRAIN_DATASET_REVISION`;
`PLAYPEN_DATASET_REVISION` controls evaluation data. Verify dataset counts when
reproducing the recipe.

The Phase-C launcher saves the generated pairs to
`artifacts/branch-<timestamp>/pairs.jsonl.gz` and records the corresponding
trajectories, then trains on those pairs. The pair file from the original
Phase-C run was not retained. Pair generation samples at temperature 0.7, so
regenerated pairs vary between runs and retraining is not expected to
reproduce the released weights exactly.

## Scale and package

Phase D has no optimizer or training data. Apply scaling **once** to the unscaled
Phase-C output, not to the already scaled adapter from Hugging Face.
Use separate new output directories.

```bash
"$PLAYPEN_VENV_DIR/bin/python" scripts/scale_lora_adapter.py \
  /absolute/path/to/C/final artifacts/beta0p20-s085 0.85
"$PLAYPEN_VENV_DIR/bin/python" scripts/merge_lora_fp32.py \
  --base Qwen/Qwen3.5-2B --adapter artifacts/beta0p20-s085 --out artifacts/merged-fp32
```

Use `--base-revision` to pin the base-model download. The merge script converts
the base weights to fp32 before applying the adapter and saves the merged
weights in fp32.

## Evaluate the submitted model

Download the submitted revision and point the plain-model registry entry at
that local directory. Downloading the full model requires approximately
7.5 GB of disk space.

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

These commands run **public-development** evaluation on the `validation` split
of [colab-potsdam/playpen-data at revision 557d8caf](https://huggingface.co/datasets/colab-potsdam/playpen-data/tree/557d8cafd1bc4557bc22803a6d4502ef53fb664c),
using `instances` for interactive games and `instances-static` for static
benchmarks. The wrapper preserves the pinned Playpen CLI's defaults of
temperature 0 and 300 generated tokens; `PLAYPEN_EVAL_TEMPERATURE` and
`PLAYPEN_EVAL_MAX_TOKENS` provide explicit overrides recorded in metadata.
The organizer's official evaluation instead used a 5,000-token limit and an
expanded public suite (72 interactive episodes rather than the development
suite's 67), plus separate closed suites. Setting a 5,000-token limit alone
does not reproduce that complete protocol or its scores.

For adapter development runs, use `PLAYPEN_QWEN_PEFT_MODEL_NAME` and
`PLAYPEN_QWEN_PEFT_ADAPTER_PATH` instead of the vanilla settings, and retain
separate results from fp32 merged-weight runs. All registry entries disable
thinking. Load the released merged checkpoint with `torch_dtype="auto"` to
retain its stored fp32 precision.

See the [model card](MODEL_CARD.md#official-final-evaluation) for official and
development results, and [configs/artifact.json](../configs/artifact.json) for
the submitted revision and model configuration.
