import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import trl
from datasets import load_dataset
from peft import PeftModel

from clemcore.clemgame import ClemGameEnv
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel

from playpen import BasePlaypenTrainer
from playpen.agents import ClemAgent, ClemObservation
from playpen.agents.openenv import ClemGameEnvAgent

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from grpo_rollout import generate_completion_with_logprobs


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


def env_csv(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


@dataclass
class GrpoEpisodeRollout:
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    env_masks: list[int] = field(default_factory=list)
    reward: float = 0.0

    def reset(self):
        self.prompt_ids.clear()
        self.completion_ids.clear()
        self.logprobs.clear()
        self.env_masks.clear()
        self.reward = 0.0


@dataclass
class GrpoEpisodeRollouts:
    prompt_ids: list[list[int]] = field(default_factory=list)
    completion_ids: list[list[int]] = field(default_factory=list)
    logprobs: list[list[float]] = field(default_factory=list)
    env_mask: list[list[int]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def append(self, rollout: GrpoEpisodeRollout):
        self.prompt_ids.append(list(rollout.prompt_ids))
        self.completion_ids.append(list(rollout.completion_ids))
        self.logprobs.append(list(rollout.logprobs))
        self.env_mask.append(list(rollout.env_masks))
        self.rewards.append(rollout.reward)

    def reset(self):
        self.prompt_ids.clear()
        self.completion_ids.clear()
        self.logprobs.clear()
        self.env_mask.clear()
        self.rewards.clear()


@dataclass
class WordleConstraintState:
    required_letters: set[str] = field(default_factory=set)
    absent_letters: set[str] = field(default_factory=set)
    fixed_positions: dict[int, str] = field(default_factory=dict)
    banned_positions: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))


def extract_guess(response: str) -> str | None:
    match = re.search(r"(?im)^guess:\s*([a-z]+)\s*$", response.strip())
    if match:
        return match.group(1).lower()
    return None


def parse_guess_feedback(observation: str) -> list[tuple[str, str]]:
    match = re.search(r"(?im)^guess_feedback:\s*(.+)$", observation)
    if not match:
        return []
    return [(letter.lower(), color.lower()) for letter, color in re.findall(r"([a-z])<(green|yellow|red)>", match.group(1))]


def update_wordle_constraints(state: WordleConstraintState, guess: str, feedback: list[tuple[str, str]]):
    if len(guess) != 5 or len(feedback) != 5:
        return

    non_red_letters = {letter for letter, color in feedback if color != "red"}
    for index, (letter, color) in enumerate(feedback):
        if color == "green":
            state.fixed_positions[index] = letter
            state.required_letters.add(letter)
        elif color == "yellow":
            state.required_letters.add(letter)
            state.banned_positions[index].add(letter)
        elif color == "red":
            state.banned_positions[index].add(letter)
            if letter not in non_red_letters and letter not in state.required_letters:
                state.absent_letters.add(letter)

    state.absent_letters.difference_update(state.required_letters)


def compute_constraint_reward(guess: str, state: WordleConstraintState) -> float:
    valid_reward = env_float("PLAYPEN_WORDLE_CONSTRAINT_VALID_REWARD", 0.0)
    missing_required_penalty = env_float("PLAYPEN_WORDLE_MISSING_REQUIRED_LETTER_PENALTY", 0.0)
    absent_letter_penalty = env_float("PLAYPEN_WORDLE_ABSENT_LETTER_PENALTY", 0.0)
    green_position_penalty = env_float("PLAYPEN_WORDLE_GREEN_POSITION_PENALTY", 0.0)
    bad_position_penalty = env_float("PLAYPEN_WORDLE_BAD_POSITION_PENALTY", 0.0)

    violations = 0
    reward = 0.0

    for index, letter in state.fixed_positions.items():
        if guess[index] != letter:
            reward -= green_position_penalty
            violations += 1

    for index, letter in enumerate(guess):
        if letter in state.banned_positions[index]:
            reward -= bad_position_penalty
            violations += 1
        if letter in state.absent_letters:
            reward -= absent_letter_penalty
            violations += 1

    for letter in state.required_letters:
        if letter not in guess:
            reward -= missing_required_penalty
            violations += 1

    if violations == 0 and (state.required_letters or state.fixed_positions or state.absent_letters):
        reward += valid_reward

    return reward


def compute_wordle_shaping(responses: list[str], env_observations: list[str]) -> float:
    schema_reward = env_float("PLAYPEN_WORDLE_SCHEMA_REWARD", 0.0)
    missing_explanation_penalty = env_float("PLAYPEN_WORDLE_MISSING_EXPLANATION_PENALTY", 0.5)
    missing_guess_penalty = env_float("PLAYPEN_WORDLE_MISSING_GUESS_PENALTY", 0.5)
    novel_guess_reward = env_float("PLAYPEN_WORDLE_NOVEL_GUESS_REWARD", 0.0)
    preamble_penalty = env_float("PLAYPEN_WORDLE_PREAMBLE_PENALTY", 0.0)
    bad_length_penalty = env_float("PLAYPEN_WORDLE_BAD_LENGTH_PENALTY", 0.0)
    repeat_guess_penalty = env_float("PLAYPEN_WORDLE_REPEAT_GUESS_PENALTY", 0.0)

    reward = 0.0
    seen_guesses = set()
    state = WordleConstraintState()
    for response in responses:
        stripped = response.strip()
        starts_with_explanation = stripped.lower().startswith("explanation:")
        if starts_with_explanation:
            reward += schema_reward
        else:
            reward -= missing_explanation_penalty
            if "guess:" in stripped.lower():
                reward -= preamble_penalty

    for turn_index, response in enumerate(responses):
        guess = extract_guess(response)
        if guess is None:
            reward -= missing_guess_penalty
        elif len(guess) != 5 or not guess.isalpha() or not guess.islower():
            reward -= bad_length_penalty
        else:
            if guess in seen_guesses:
                reward -= repeat_guess_penalty
            else:
                reward += novel_guess_reward
                seen_guesses.add(guess)

            if turn_index > 0:
                reward += compute_constraint_reward(guess, state)

        if guess is not None and turn_index + 1 < len(env_observations):
            feedback = parse_guess_feedback(env_observations[turn_index + 1])
            if feedback:
                update_wordle_constraints(state, guess, feedback)

    return reward


def compute_wordle_rollout_reward(responses: list[str], env_observations: list[str], terminal_reward: float) -> float:
    reward = terminal_reward + compute_wordle_shaping(responses, env_observations)
    abort_jitter = env_float("PLAYPEN_ABORT_JITTER", 0.0)
    if math.isclose(terminal_reward, -1.0, abs_tol=1e-6) and abort_jitter > 0:
        reward -= random.uniform(0.0, abort_jitter)
    return reward


class WordleGrpoAgent(ClemAgent):

    def __init__(self, trainer: trl.GRPOTrainer):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = trainer.processing_class
        self.episode = GrpoEpisodeRollout()
        self.responses: list[str] = []
        self.env_observations: list[str] = []
        self._first_turn = True
        self.max_new_tokens = int(getattr(trainer.args, "max_completion_length", env_int("PLAYPEN_MAX_COMPLETION_LENGTH", 1024)))
        self.temperature = float(getattr(trainer.args, "temperature", env_float("PLAYPEN_TEMPERATURE", 0.7)))
        self.top_p = env_float("PLAYPEN_TOP_P", 0.95)

    def track_env_completion(self, last: ClemObservation):
        self.env_observations.append(last.content)
        if self._first_turn:
            prompt_text = self.tokenizer.apply_chat_template(self.history, add_generation_prompt=True, tokenize=False)
            self.episode.prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            self._first_turn = False
            return
        env_feedback_ids = self.tokenizer.encode("\n\n" + last.content, add_special_tokens=False)
        self.episode.completion_ids.extend(env_feedback_ids)
        self.episode.logprobs.extend([0.0] * len(env_feedback_ids))
        self.episode.env_masks.extend([0] * len(env_feedback_ids))

    def track_agent_completion(self, outputs: dict, response: str):
        self.episode.completion_ids.extend(outputs["completion_ids"])
        self.episode.logprobs.extend(outputs["logprobs"])
        self.episode.env_masks.extend([1] * len(outputs["completion_ids"]))
        self.responses.append(response)

    def act(self, last: ClemObservation) -> str:
        self.track_env_completion(last)
        prompt_text = self.tokenizer.apply_chat_template(self.history, add_generation_prompt=True, tokenize=False)
        rollout_model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        outputs = generate_completion_with_logprobs(
            model=rollout_model,
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        response = outputs.get("text") or self.tokenizer.decode(outputs["completion_ids"], skip_special_tokens=True)
        self.track_agent_completion(outputs, response)
        return response

    def get_episode(self) -> GrpoEpisodeRollout:
        return self.episode

    def reset(self):
        super().reset()
        self.episode.reset()
        self.responses.clear()
        self.env_observations.clear()
        self._first_turn = True


class Qwen35WordleGrpoTrainer(BasePlaypenTrainer):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        base_url = os.getenv("PLAYPEN_OPENENV_BASE_URL", "http://127.0.0.1:9000")
        self.game_env = ClemGameEnv(base_url=base_url).sync()

    def rollout_episode(self, env: ClemGameEnv, agent: ClemGameEnvAgent, prompt: str) -> GrpoEpisodeRollout:
        obs = env.reset(game_id=int(prompt))
        while not obs.done:
            action = agent(obs)
            obs = env.step(action)
        rollout = agent.wrapped_agent.get_episode()
        rollout.reward = compute_wordle_rollout_reward(
            agent.wrapped_agent.responses,
            agent.wrapped_agent.env_observations,
            float(obs.reward),
        )
        return rollout

    def rollout_func(self, prompts: list[str], trainer: trl.GRPOTrainer) -> dict:
        agent = ClemGameEnvAgent(WordleGrpoAgent(trainer))
        rollouts = GrpoEpisodeRollouts()
        try:
            for prompt in prompts:
                rollout = self.rollout_episode(self.game_env, agent, str(prompt))
                rollouts.append(rollout)
                agent.reset()
            return asdict(rollouts)
        finally:
            rollouts.reset()
            agent.reset()

    @staticmethod
    def reward_env(completions: list[str], **kwargs) -> list[float]:
        return kwargs.get("rewards", [0.0] * len(completions))

    def learn(self):
        seed = env_int("PLAYPEN_SEED", 42)
        random.seed(seed)

        dataset = load_dataset("colab-potsdam/playpen-data", "instances", split="train")
        dataset = dataset.filter(lambda game_instance: game_instance["game"] == "wordle")

        experiments_filter = set(env_csv("PLAYPEN_EXPERIMENTS"))
        if experiments_filter:
            dataset = dataset.filter(lambda row: row["experiment"] in experiments_filter)

        task_ids_filter = {int(x) for x in env_csv("PLAYPEN_TASK_IDS")}
        if task_ids_filter:
            dataset = dataset.filter(lambda row: row["task_id"] in task_ids_filter)

        max_instances = env_int("PLAYPEN_MAX_INSTANCES", 0)
        if max_instances > 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(max_instances, len(dataset))))

        if len(dataset) == 0:
            raise ValueError("No wordle RL instances remain after filtering.")

        dataset = dataset.add_column("prompt", [str(x) for x in dataset["task_id"]])

        output_dir = os.getenv("PLAYPEN_OUTPUT_DIR", f"models/grpo+lora/{self.learner.name}-wordle")
        learning_rate = env_float("PLAYPEN_LR", 1e-6)
        num_epochs = env_float("PLAYPEN_NUM_TRAIN_EPOCHS", 1.0)
        batch_size = env_int("PLAYPEN_BATCH_SIZE", 1)
        grad_accum = env_int("PLAYPEN_GRAD_ACCUM", 4)
        num_generations = env_int("PLAYPEN_NUM_GENERATIONS", 4)
        generation_batch_size = env_int("PLAYPEN_GENERATION_BATCH_SIZE", max(batch_size, num_generations))
        temperature = env_float("PLAYPEN_TEMPERATURE", 0.7)
        max_completion_length = env_int("PLAYPEN_MAX_COMPLETION_LENGTH", 1024)
        vllm_gpu_mem = env_float("PLAYPEN_VLLM_GPU_MEM", 0.35)
        beta = env_float("PLAYPEN_GRPO_BETA", 0.02)
        logging_steps = env_int("PLAYPEN_LOGGING_STEPS", 1)
        save_steps = env_int("PLAYPEN_SAVE_STEPS", 20)
        save_total_limit = env_int("PLAYPEN_SAVE_TOTAL_LIMIT", 2)
        gradient_checkpointing = env_flag("PLAYPEN_GRADIENT_CHECKPOINTING", False)
        disable_dropout = env_flag("PLAYPEN_DISABLE_DROPOUT", True)

        self.learner.model.config.use_cache = not gradient_checkpointing
        is_existing_peft = isinstance(self.learner.model, PeftModel)
        if is_existing_peft:
            print("Continuing GRPO training from an existing PEFT adapter.")
            adapter_name = getattr(self.learner.model, "active_adapter", "default")
            if isinstance(adapter_name, list):
                adapter_name = adapter_name[0]
            self.learner.model.set_adapter(adapter_name)
            self.learner.model.train()
            self.learner.model.print_trainable_parameters()

        print("Wordle RL instances:", len(dataset))
        print("Task ids:", sorted(dataset["task_id"]))
        print("Generation batch size:", generation_batch_size)

        if generation_batch_size % num_generations != 0:
            raise ValueError(
                f"PLAYPEN_GENERATION_BATCH_SIZE ({generation_batch_size}) must be divisible by "
                f"PLAYPEN_NUM_GENERATIONS ({num_generations})."
            )

        config = trl.GRPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
            bf16=True,
            seed=seed,
            disable_dropout=disable_dropout,
            report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_strategy="steps",
            save_total_limit=save_total_limit,
            num_generations=num_generations,
            generation_batch_size=generation_batch_size,
            max_completion_length=max_completion_length,
            temperature=temperature,
            use_vllm=env_flag("PLAYPEN_USE_VLLM", False),
            vllm_mode=os.getenv("PLAYPEN_VLLM_MODE", "colocate"),
            vllm_gpu_memory_utilization=vllm_gpu_mem,
            beta=beta,
            log_completions=env_flag("PLAYPEN_LOG_COMPLETIONS", False),
        )

        trainer = trl.GRPOTrainer(
            model=self.learner.model,
            reward_funcs=self.reward_env,
            train_dataset=dataset,
            processing_class=self.learner.tokenizer,
            args=config,
            peft_config=None,
            rollout_func=self.rollout_func,
        )

        self.game_env.connect()
        try:
            trainer.train()
            final_dir = os.path.join(output_dir, "final")
            trainer.save_model(final_dir)
            print("Saved final adapter to:", final_dir)
        finally:
            self.game_env.close()
