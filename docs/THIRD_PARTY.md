# Third-party sources and attribution

Project additions are distributed under the root [Apache-2.0 LICENSE](../LICENSE).
This does not replace the terms of upstream software, model weights or datasets.

- **Playpen**: [lm-playpen/playpen](https://github.com/lm-playpen/playpen),
  reference commit `b69702bf32ea50a1e74ec4ead288b242adfb0ae8`. The trainers build
  on its interfaces and examples; `trainers/data_utils.py` derives from its
  example helper. The evaluation patch modifies that upstream CLI. The
  [MIT notice](../LICENSES/playpen-MIT.txt) is retained here; the installed
  upstream checkout also retains its LICENSE.
- **clembench / clemcore**: installed from their upstream repositories/packages,
  retaining their own licenses and game resource notices. The development
  game-tree pin is `ed3948604695b077d285654b73fcd8aae7a12bff`, with static overlay
  `b716d279ae70befc988bffff2fb1499ccc81866a`.
- **Qwen3.5-2B**: base model from [Qwen](https://huggingface.co/Qwen/Qwen3.5-2B).
  Model weights are hosted on Hugging Face and not included in this repository.
- **Playpen data**: loaded from
  [colab-potsdam/playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data).
  Dataset and resource terms remain attached to those upstream materials.

This release does not bundle source datasets, base weights, evaluator game
trees or hidden test data. Generated research outputs should retain source
provenance and be reviewed separately before redistribution.
