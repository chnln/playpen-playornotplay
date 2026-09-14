# Development comparisons

We compared successive checkpoints during development and in a separate
controlled campaign. The changes in Clemscore and Statscore varied by phase
and evaluation setting:

| Comparison | Pinned development trajectory | Separate controlled campaign |
| --- | ---: | ---: |
| Turn-DPO pass 1 | +2.98 clem | +1.49 clem |
| Turn-DPO pass 2 | −1.31 clem | +1.34 clem |
| Branch-DPO | +3.00 clem | −2.07 clem / +1.23 stat |

These are different evaluation settings, not an additive decomposition of the
official final score. Sequential checkpoint changes do not isolate causal
contributions.

The late-continuation campaign evaluated ten endpoints: eight DPO runs and
two training-free rescalings of a preceding DPO run. Five DPO runs started
from the selected `s=0.85` adapter; three used Phase-A or unscaled Phase-C
parents. The five runs from the selected adapter scored 44.15–48.92 Clemscore,
below its 50.43 development reference. These experiments did not change the
submitted model. Whole-dialogue preference experiments and 4B SFT probes were
separate exploratory work.

The weak-game SFT and Wordle GRPO runs applied no parameter updates. Their
scores therefore provide no evidence about the effectiveness of those
training methods.

The submitted model approximately preserves aggregate Statscore, although
individual static benchmark components do not all improve. See the
[model card](MODEL_CARD.md) for official scores and the submitted revision,
and the [paper](https://arxiv.org/abs/2608.28458v1) for data construction and
reproducibility details.
