import os
import re
from collections import Counter

import trl
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel

from playpen import BasePlaypenTrainer


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


def env_csv(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


WORDLE_GUESS_RE = re.compile(r"^guess:\s*([a-z]+)\s*$", re.MULTILINE)


def is_clean_wordle_episode(episode: dict) -> bool:
    if episode["meta"]["game"] != "wordle":
        return True

    guesses: list[str] = []
    invalid_guess = False
    for message in episode["messages"]:
        if message["role"] == "assistant":
            match = WORDLE_GUESS_RE.search(message["content"])
            if match:
                guesses.append(match.group(1))
        elif "not a valid word" in message["content"].lower():
            invalid_guess = True

    return not invalid_guess and len(guesses) == len(set(guesses))


def explode_conversations_on_assistant_turns(
    dataset,
    min_assistant_turn: int,
    required_prev_user_substring: str,
):
    rows = []
    for episode in dataset:
        assistant_turn = 0
        prefix_messages = []
        previous_user_content = ""
        for message in episode["messages"]:
            copied = {"role": message["role"], "content": message["content"]}
            prefix_messages.append(copied)
            if message["role"] == "user":
                previous_user_content = message["content"]
                continue

            assistant_turn += 1
            if assistant_turn < min_assistant_turn:
                continue
            if required_prev_user_substring and required_prev_user_substring not in previous_user_content:
                continue

            rows.append({
                "messages": list(prefix_messages),
                "meta": dict(episode["meta"]),
                "assistant_turn_index": assistant_turn,
            })

    return Dataset.from_list(rows)


class Qwen35LoraSftTrainer(BasePlaypenTrainer):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)

    def learn(self):
        local_data = env_str("PLAYPEN_LOCAL_DATA", "")
        if local_data:
            dataset = load_dataset("json", data_files=local_data, split="train")
            print(f"Loaded local data from {local_data}: {len(dataset)} episodes")
        else:
            dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train",
                                   revision=os.getenv("PLAYPEN_TRAIN_DATASET_REVISION") or None)
        seed = env_int("PLAYPEN_SEED", 42)
        explode_assistant_turns = env_flag("PLAYPEN_EXPLODE_ASSISTANT_TURNS", False)
        min_assistant_turn = env_int("PLAYPEN_MIN_ASSISTANT_TURN", 1)
        required_prev_user_substring = env_str("PLAYPEN_REQUIRE_PREV_USER_SUBSTRING", "")

        outcomes = set(env_csv("PLAYPEN_OUTCOMES"))
        if outcomes:
            dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] in outcomes)
        elif env_flag("PLAYPEN_SUCCESS_ONLY", True):
            dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")

        games_filter = set(env_csv("PLAYPEN_GAMES"))
        if games_filter:
            dataset = dataset.filter(lambda episode: episode["meta"]["game"] in games_filter)

        roles_filter = set(env_csv("PLAYPEN_ROLES"))
        if roles_filter:
            dataset = dataset.filter(lambda episode: episode["meta"]["game_role"] in roles_filter)

        source_models_filter = set(env_csv("PLAYPEN_SOURCE_MODELS"))
        if source_models_filter:
            dataset = dataset.filter(lambda episode: episode["meta"]["model"] in source_models_filter)

        if env_flag("PLAYPEN_WORDLE_CLEAN_ONLY", False):
            dataset = dataset.filter(is_clean_wordle_episode)

        max_samples_per_game = env_int("PLAYPEN_MAX_SAMPLES_PER_GAME", 0)
        if max_samples_per_game > 0:
            dataset = dataset.shuffle(seed=seed)
            per_game_counts = Counter()
            keep_indices = []
            for index, episode in enumerate(dataset):
                game = episode["meta"]["game"]
                if per_game_counts[game] >= max_samples_per_game:
                    continue
                per_game_counts[game] += 1
                keep_indices.append(index)
            dataset = dataset.select(keep_indices)

        if explode_assistant_turns:
            dataset = explode_conversations_on_assistant_turns(
                dataset,
                min_assistant_turn=min_assistant_turn,
                required_prev_user_substring=required_prev_user_substring,
            )
            print("Expanded to assistant-turn samples:", len(dataset))

        max_samples = env_int("PLAYPEN_MAX_TRAIN_SAMPLES", 0)
        if max_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

        game_counts = Counter(episode["meta"]["game"] for episode in dataset)
        role_counts = Counter(episode["meta"]["game_role"] for episode in dataset)
        source_model_counts = Counter(episode["meta"]["model"] for episode in dataset)
        print("Filtered dataset size:", len(dataset))
        print("Top games:", game_counts.most_common(10))
        print("Top roles:", role_counts.most_common(10))
        print("Top source models:", source_model_counts.most_common(10))
        if explode_assistant_turns and len(dataset) > 0 and "assistant_turn_index" in dataset.column_names:
            assistant_turn_counts = Counter(int(turn) for turn in dataset["assistant_turn_index"])
            print("Assistant turn indices:", sorted(assistant_turn_counts.items()))

        eval_ratio = env_float("PLAYPEN_EVAL_RATIO", 0.1)
        dataset = dataset.train_test_split(test_size=eval_ratio, shuffle=True, seed=seed)

        output_dir = os.getenv("PLAYPEN_OUTPUT_DIR", f"models/sft+lora/{self.learner.name}-success")
        max_length = env_int("PLAYPEN_MAX_LENGTH", 2048)
        batch_size = env_int("PLAYPEN_BATCH_SIZE", 2)
        grad_accum = env_int("PLAYPEN_GRAD_ACCUM", 8)
        learning_rate = env_float("PLAYPEN_LR", 2e-4)
        num_epochs = env_float("PLAYPEN_NUM_TRAIN_EPOCHS", 1.0)
        save_steps = env_int("PLAYPEN_SAVE_STEPS", 100)
        eval_steps = env_int("PLAYPEN_EVAL_STEPS", 100)
        logging_steps = env_int("PLAYPEN_LOGGING_STEPS", 10)
        save_total_limit = env_int("PLAYPEN_SAVE_TOTAL_LIMIT", 2)

        self.learner.model.config.use_cache = False
        is_existing_peft = isinstance(self.learner.model, PeftModel)
        if is_existing_peft:
            print("Continuing training from an existing PEFT adapter.")
            adapter_name = getattr(self.learner.model, "active_adapter", "default")
            if isinstance(adapter_name, list):
                adapter_name = adapter_name[0]
            self.learner.model.set_adapter(adapter_name)
            self.learner.model.train()
            self.learner.model.print_trainable_parameters()

        config = trl.SFTConfig(
            max_length=max_length,
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            packing=False,
            completion_only_loss=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
        )

        trainer = trl.SFTTrainer(
            model=self.learner.model,
            processing_class=self.learner.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=config,
            peft_config=None
            if is_existing_peft
            else LoraConfig(
                r=env_int("PLAYPEN_LORA_R", 16),
                lora_alpha=env_int("PLAYPEN_LORA_ALPHA", 32),
                lora_dropout=env_float("PLAYPEN_LORA_DROPOUT", 0.05),
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )

        trainer.train()
        final_dir = os.path.join(output_dir, "final")
        trainer.save_model(final_dir)
        print("Saved final adapter to:", final_dir)
