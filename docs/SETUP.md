# Setup and runtime provenance

Use a separate Python 3.12 environment on a Linux NVIDIA machine. The canonical
training configurations target one A100-80GB GPU; the SLURM partition defaults
to `gpua100` and can be changed with `PLAYPEN_SLURM_PARTITION`. Do not run training
on a login node or expect these batch sizes to fit a smaller GPU unchanged.

The reference dependencies in [requirements-training.txt](../requirements-training.txt)
include torch 2.6.0, transformers 5.4.0, PEFT 0.18.1 and TRL 0.29.1, based on
the recorded training environment. The April environment used clemcore 3.7.1;
these requirements use 3.7.2, the version used by the organizer. The file pins
direct dependencies only; transitive dependencies and CUDA are not locked.
Check driver/CUDA compatibility on the target machine before training.

## Install the pinned upstream workspace

Run these commands from this repository's root in a fresh checkout. They create
ignored local directories; they do not install into system Python. `uv` must
already be available. Existing upstream directories should be inspected rather
than overwritten or reset.

```bash
git clone https://github.com/lm-playpen/playpen.git environment/playpen
git -C environment/playpen checkout b69702bf32ea50a1e74ec4ead288b242adfb0ae8
git -C environment/playpen apply --check ../../patches/playpen-dataset-revision.patch
git -C environment/playpen apply ../../patches/playpen-dataset-revision.patch

git clone https://github.com/clp-research/clembench.git environment/playpen/clembench
git -C environment/playpen/clembench checkout ed3948604695b077d285654b73fcd8aae7a12bff
git -C environment/playpen/clembench restore \
  --source b716d279ae70befc988bffff2fb1499ccc81866a --worktree -- static

uv venv --python 3.12 environment/playpen/venv_repro
uv pip install --python environment/playpen/venv_repro/bin/python \
  -r requirements-training.txt \
  -r environment/playpen/clembench/requirements.txt \
  -e 'environment/playpen[trl]'
uv pip check --python environment/playpen/venv_repro/bin/python
```

The CLI patch makes `PLAYPEN_DATASET_REVISION` affect public evaluation.
Training uses `PLAYPEN_TRAIN_DATASET_REVISION` in our three trainers. The two
settings independently pin the training and evaluation dataset snapshots.

The clembench checkout intentionally combines the historical interactive pin
with a newer static directory for clemcore compatibility. Its HEAD stays at
`ed39486`; `git diff HEAD -- static` records the overlay. These revisions define
our development setup. The organizer did not record the exact game-tree
revision used for the official evaluation.
Review any game-specific resource requirements in the upstream checkout.

```bash
export PLAYPEN_VENV_DIR="$PWD/environment/playpen/venv_repro"
export CLEMBENCH_HOME="$PWD/environment/playpen"
"$PLAYPEN_VENV_DIR/bin/playpen" list games
"$PLAYPEN_VENV_DIR/bin/python" scripts/check_torch_cuda.py --require-gpu
"$PLAYPEN_VENV_DIR/bin/wandb" login
```

Run GPU diagnostics on the allocated compute node. `wandb` must be installed and
authenticated before training; the recipe launcher uses project
`playpen-wordle` and a distinct run name. No credential belongs in this repo.

## Lightweight validation

These checks need only Python 3.12 and Bash; they do not load a model or submit
a GPU job:

```bash
"$PLAYPEN_VENV_DIR/bin/python" -m unittest discover -s tests
"$PLAYPEN_VENV_DIR/bin/python" scripts/submit_recipe_phase.py A --dry-run
```

The offline tests cover phase selection, input validation, model registration
and fp32/EOS packaging with mocked models. They do not cover dependency
resolution, CUDA, training or complete benchmark runs. Test the installation
with a small SLURM training and evaluation run before starting full experiments.
Keep results and `run_metadata.json`; scores depend on the serving environment
and evaluation protocol as well as the model weights.
