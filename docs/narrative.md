# Method overview

We fine-tuned Qwen3.5-2B through broad SFT (A), two Wordle turn-DPO passes (B),
and Wordle branch-DPO (C), then scaled the LoRA delta and merged in fp32 (D).
The weak-game SFT and Wordle GRPO runs applied no parameter updates and did
not contribute to the evaluated weights.

Broad SFT produces the largest improvement in the development trajectory.
Wordle provides mechanically checkable post-feedback decisions for turn-local
repair; branch-DPO uses the model's own trajectories and downstream scores.
Only turn-DPO pass 1 has a positive Clemscore change in both evaluation
settings; pass 2 and branch-DPO have changes with opposite signs across them.
These [checkpoint comparisons](negative_results.md) do not isolate a causal
contribution for each phase. Delta scaling selects the highest aggregate
Statscore subject to a minimum Clemscore on the public-development set.

The submitted model achieved 38.92 public Clemscore and 44.14 Statscore, with
41.17 closed in-domain and 7.88 closed out-of-domain Clemscore. Out-of-domain
transfer remains limited, with the largest gains concentrated in held-out
Wordle variants. Development measurements used a different evaluation setup.
See the [model card](MODEL_CARD.md)
for evaluation protocols and training configuration, the
[paper](https://arxiv.org/abs/2608.28458v1) for the detailed methodology, and
the [command guide](TRAINING_PROCESS.md) to run the pipeline.
