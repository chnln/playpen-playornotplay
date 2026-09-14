import copy
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import trl
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel

from playpen import BasePlaypenTrainer

TRL_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(TRL_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_EXAMPLES_DIR))

from wordle_solver import expected_reduction_score, top_k_guesses


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


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


WORDLE_GUESS_RE = re.compile(r"^guess:\s*([a-z]+)\s*$", re.MULTILINE)
WORDLE_EXPLANATION_RE = re.compile(r"^explanation:\s*(.*?)\s*$", re.MULTILINE)
WORDLE_FEEDBACK_RE = re.compile(r"([a-zA-Z])<([a-zA-Z]+)>")


@dataclass
class WordleConstraintState:
    required_letters: set[str] = field(default_factory=set)
    banned_letters: set[str] = field(default_factory=set)
    fixed_positions: dict[int, str] = field(default_factory=dict)
    banned_positions: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))


def extract_guess(text: str) -> str | None:
    match = WORDLE_GUESS_RE.search(text)
    if not match:
        return None
    return match.group(1)


def replace_guess(text: str, new_guess: str) -> str | None:
    if not WORDLE_GUESS_RE.search(text):
        return None
    return WORDLE_GUESS_RE.sub(f"guess: {new_guess}", text, count=1)


def render_solver_teacher_reply(text: str, new_guess: str) -> str | None:
    if not WORDLE_GUESS_RE.search(text):
        return None

    updated = WORDLE_GUESS_RE.sub(f"guess: {new_guess}", text, count=1)
    if WORDLE_EXPLANATION_RE.search(updated):
        return WORDLE_EXPLANATION_RE.sub(
            "explanation: I am choosing a valid candidate that matches the current feedback and should reduce the remaining possibilities.",
            updated,
            count=1,
        )
    return (
        "explanation: I am choosing a valid candidate that matches the current feedback and should reduce the remaining possibilities.\n"
        f"guess: {new_guess}"
    )


def parse_feedback(text: str) -> list[tuple[str, str]] | None:
    if "guess_feedback" not in text.lower():
        return None
    pairs = [(m.group(1).lower(), m.group(2).lower()) for m in WORDLE_FEEDBACK_RE.finditer(text)]
    return pairs or None


def update_constraint_state(state: WordleConstraintState, guess: str, feedback: list[tuple[str, str]]) -> None:
    if not feedback or len(feedback) != len(guess):
        return

    present_letters = {letter for letter, color in feedback if color in {"green", "yellow"}}
    for idx, (letter, color) in enumerate(feedback):
        if color == "green":
            state.required_letters.add(letter)
            state.fixed_positions[idx] = letter
        elif color == "yellow":
            state.required_letters.add(letter)
            state.banned_positions[letter].add(idx)
        elif color == "red":
            if letter in present_letters:
                state.banned_positions[letter].add(idx)
            else:
                state.banned_letters.add(letter)


def word_matches_state(word: str, state: WordleConstraintState) -> bool:
    if len(word) != 5 or not word.isalpha():
        return False
    for idx, letter in state.fixed_positions.items():
        if word[idx] != letter:
            return False
    for idx, letter in enumerate(word):
        if letter in state.banned_letters:
            return False
        if idx in state.banned_positions.get(letter, set()):
            return False
    if any(letter not in word for letter in state.required_letters):
        return False
    return True


def remaining_valid_words(
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    chosen_guess: str | None = None,
) -> list[str]:
    blocked = set(prior_guesses)
    if chosen_guess:
        blocked.add(chosen_guess)
    return [
        word
        for word in valid_words
        if word not in blocked and word_matches_state(word, state)
    ]


def candidate_quality_scores(candidates: list[str]) -> dict[str, float]:
    if not candidates:
        return {}

    unique_letter_counts = Counter()
    position_letter_counts = [Counter() for _ in range(5)]
    for word in candidates:
        for letter in set(word):
            unique_letter_counts[letter] += 1
        for idx, letter in enumerate(word):
            position_letter_counts[idx][letter] += 1

    scale = float(len(candidates))
    scores: dict[str, float] = {}
    for word in candidates:
        unique_letters = set(word)
        letter_coverage = sum(unique_letter_counts[letter] for letter in unique_letters) / scale
        position_coverage = sum(position_letter_counts[idx][letter] for idx, letter in enumerate(word)) / scale
        repeat_penalty = float(len(word) - len(unique_letters))
        scores[word] = letter_coverage + position_coverage - (2.0 * repeat_penalty)
    return scores


def select_low_info_valid_guess(
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    chosen_guess: str,
    min_gap: float,
) -> str | None:
    candidates = remaining_valid_words(valid_words, state, prior_guesses, chosen_guess=chosen_guess)
    if not candidates:
        return None

    scored_words = candidate_quality_scores(candidates + [chosen_guess])
    chosen_score = scored_words.get(chosen_guess)
    if chosen_score is None:
        return None

    ranked = sorted(
        ((word, score) for word, score in scored_words.items() if word != chosen_guess),
        key=lambda item: (item[1], item[0]),
    )
    if not ranked:
        return None

    rejected_word, rejected_score = ranked[0]
    if chosen_score - rejected_score < min_gap:
        return None
    return rejected_word


def select_trap_guess(
    trap_guesses: list[str],
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    chosen_guess: str,
) -> str | None:
    blocked = set(prior_guesses)
    blocked.add(chosen_guess)
    for guess in trap_guesses:
        if guess in blocked:
            continue
        if valid_words and guess not in valid_words:
            continue
        if word_matches_state(guess, state):
            return guess
    return None


def select_invalid_length_trap_guess(
    trap_guesses: list[str],
    prior_guesses: list[str],
    chosen_guess: str,
) -> str | None:
    blocked = set(prior_guesses)
    blocked.add(chosen_guess)
    for guess in trap_guesses:
        guess = guess.strip().lower()
        if not guess or guess in blocked:
            continue
        if not guess.isalpha():
            continue
        if len(guess) == 5:
            continue
        return guess
    return None


def select_solver_teacher_guess(
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    chosen_guess: str,
    solver_top_k: int,
    solver_max_exact_pool: int,
    min_solver_gain: float,
) -> str | None:
    candidates = remaining_valid_words(valid_words, state, prior_guesses)
    if not candidates:
        return None

    ranked = top_k_guesses(
        candidates,
        k=min(max(1, solver_top_k), len(candidates)),
        max_exact_pool=solver_max_exact_pool,
    )
    if not ranked:
        return None

    solver_guess = ranked[0]
    if solver_guess == chosen_guess:
        return None

    if chosen_guess in candidates:
        solver_gain = expected_reduction_score(solver_guess, candidates)
        chosen_gain = expected_reduction_score(chosen_guess, candidates)
        if solver_gain - chosen_gain < min_solver_gain:
            return None

    return solver_guess


def _base_candidate_pool(
    valid_words: set[str],
    prior_guesses: list[str],
    chosen_guess: str,
) -> list[str]:
    blocked = set(prior_guesses)
    blocked.add(chosen_guess)
    return [
        word
        for word in valid_words
        if word not in blocked and len(word) == 5 and word.isalpha()
    ]


def select_missing_required_valid_guess(
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    chosen_guess: str,
) -> str | None:
    if not state.required_letters:
        return None

    candidates: list[tuple[str, int]] = []
    for word in _base_candidate_pool(valid_words, prior_guesses, chosen_guess):
        if any(word[idx] != letter for idx, letter in state.fixed_positions.items()):
            continue
        if any(letter in state.banned_letters for letter in word):
            continue
        if any(idx in state.banned_positions.get(letter, set()) for idx, letter in enumerate(word)):
            continue

        missing_required = sum(letter not in word for letter in state.required_letters)
        if missing_required <= 0:
            continue
        candidates.append((word, missing_required))

    if not candidates:
        return None

    quality_scores = candidate_quality_scores([word for word, _ in candidates])
    ranked = sorted(
        candidates,
        key=lambda item: (item[1], -quality_scores.get(item[0], 0.0), item[0]),
    )
    return ranked[0][0]


def select_used_red_valid_guess(
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    chosen_guess: str,
) -> str | None:
    if not state.banned_letters:
        return None

    candidates: list[tuple[str, int]] = []
    for word in _base_candidate_pool(valid_words, prior_guesses, chosen_guess):
        if any(word[idx] != letter for idx, letter in state.fixed_positions.items()):
            continue
        if any(idx in state.banned_positions.get(letter, set()) for idx, letter in enumerate(word)):
            continue
        if any(letter not in word for letter in state.required_letters):
            continue

        banned_hits = sum(letter in state.banned_letters for letter in set(word))
        if banned_hits <= 0:
            continue
        candidates.append((word, banned_hits))

    if not candidates:
        return None

    quality_scores = candidate_quality_scores([word for word, _ in candidates])
    ranked = sorted(
        candidates,
        key=lambda item: (item[1], -quality_scores.get(item[0], 0.0), item[0]),
    )
    return ranked[0][0]


def select_bad_yellow_valid_guess(
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    chosen_guess: str,
) -> str | None:
    if not state.banned_positions:
        return None

    candidates: list[tuple[str, int]] = []
    for word in _base_candidate_pool(valid_words, prior_guesses, chosen_guess):
        if any(word[idx] != letter for idx, letter in state.fixed_positions.items()):
            continue
        if any(letter in state.banned_letters for letter in word):
            continue
        if any(letter not in word for letter in state.required_letters):
            continue

        yellow_hits = sum(idx in state.banned_positions.get(letter, set()) for idx, letter in enumerate(word))
        if yellow_hits <= 0:
            continue
        candidates.append((word, yellow_hits))

    if not candidates:
        return None

    quality_scores = candidate_quality_scores([word for word, _ in candidates])
    ranked = sorted(
        candidates,
        key=lambda item: (item[1], -quality_scores.get(item[0], 0.0), item[0]),
    )
    return ranked[0][0]


def default_valid_words_file() -> str:
    return "clembench/wordle/resources/target_words/en/official_recognized_words.txt"


def default_instances_file() -> str:
    return "clembench/wordle/in/instances.json"


def candidate_instances_files() -> list[str]:
    return [
        "clembench/wordle/in/instances.json",
        "clembench/wordle/in/instances_v3.0.json",
        "clembench/wordle/in/instances_v2.0.json",
        "clembench/wordle/in/instances_v1.6.json",
        "clembench/wordle/in/instances_v1.0.json",
        "clembench/wordle/in/instances_v0.9.json",
    ]


def load_valid_words(path_str: str) -> set[str]:
    path = Path(path_str)
    if path.exists():
        words = {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if words:
            return words

    for fallback_str in candidate_instances_files():
        fallback = Path(fallback_str)
        if not fallback.exists():
            continue
        data = json.loads(fallback.read_text(encoding="utf-8"))
        words: set[str] = set()
        for experiment in data.get("experiments", []):
            for word in experiment.get("english_words", []):
                if isinstance(word, str) and word:
                    words.add(word.lower())
        if words:
            return words

    return set()


def make_invalid_word(guess: str, valid_words: set[str]) -> str:
    candidates = []

    for idx, replacement in (
        (4, "q"),
        (0, "q"),
        (2, "x"),
        (1, "z"),
        (3, "j"),
    ):
        if 0 <= idx < len(guess):
            candidate = guess[:idx] + replacement + guess[idx + 1 :]
            candidates.append(candidate)

    candidates.extend(
        [
            guess[:4] + "z",
            guess[:2] + "qq" + guess[4:5],
            "zzzzz",
            "qqqqq",
        ]
    )

    for candidate in candidates:
        if len(candidate) == 5 and candidate.isalpha() and candidate.lower() not in valid_words:
            return candidate.lower()

    # Fallback should still be invalid for Wordle-format purposes even if the list is missing.
    return "zzzzz"


def is_clean_wordle_episode(episode: dict) -> bool:
    guesses: list[str] = []
    invalid_guess = False
    for message in episode["messages"]:
        if message["role"] == "assistant":
            guess = extract_guess(message["content"])
            if guess:
                guesses.append(guess)
        elif "not a valid word" in message["content"].lower():
            invalid_guess = True

    return not invalid_guess and len(guesses) == len(set(guesses))


def build_wordle_turn_preferences(
    dataset,
    seed: int,
    source_models_filter: set[str],
    negative_modes: list[str],
    min_assistant_turn: int,
    max_assistant_turn: int,
    required_prev_user_substring: str,
    clean_only: bool,
    valid_words: set[str],
    sample_one_negative: bool,
    min_negative_gap: float,
    trap_guesses: list[str],
    long_trap_guesses: list[str],
    solver_top_k: int,
    solver_max_exact_pool: int,
    min_solver_gain: float,
):
    rng = random.Random(seed)
    rows = []
    pair_counts = Counter()
    source_counts = Counter()
    turn_counts = Counter()
    negative_kind_counts = Counter()

    for episode in dataset:
        meta = episode["meta"]
        if meta["game"] != "wordle" or meta["outcome"] != "success":
            continue
        if source_models_filter and meta["model"] not in source_models_filter:
            continue
        if clean_only and not is_clean_wordle_episode(episode):
            continue

        source_counts[meta["model"]] += 1

        history = []
        prior_guesses: list[str] = []
        previous_user_content = ""
        assistant_turn = 0
        state = WordleConstraintState()

        for message in episode["messages"]:
            if message["role"] == "user":
                history.append(dict(message))
                previous_user_content = message["content"]
                if prior_guesses:
                    feedback = parse_feedback(previous_user_content)
                    if feedback:
                        update_constraint_state(state, prior_guesses[-1], feedback)
                continue

            assistant_turn += 1
            current_assistant = dict(message)
            chosen_guess = extract_guess(current_assistant["content"])

            if (
                chosen_guess
                and assistant_turn >= min_assistant_turn
                and (max_assistant_turn <= 0 or assistant_turn <= max_assistant_turn)
                and (not required_prev_user_substring or required_prev_user_substring in previous_user_content)
            ):
                prompt = [dict(m) for m in history]
                rejected_variants: list[tuple[str, str]] = []

                if "solver_teacher" in negative_modes:
                    solver_guess = select_solver_teacher_guess(
                        valid_words=valid_words,
                        state=state,
                        prior_guesses=prior_guesses,
                        chosen_guess=chosen_guess,
                        solver_top_k=solver_top_k,
                        solver_max_exact_pool=solver_max_exact_pool,
                        min_solver_gain=min_solver_gain,
                    )
                    if solver_guess:
                        solver_chosen = render_solver_teacher_reply(current_assistant["content"], solver_guess)
                        if solver_chosen and solver_chosen != current_assistant["content"]:
                            rows.append(
                                {
                                    "prompt": prompt,
                                    "chosen": [{"role": "assistant", "content": solver_chosen}],
                                    "rejected": [current_assistant],
                                    "game": meta["game"],
                                    "game_role": meta["game_role"],
                                    "chosen_model": meta["model"],
                                    "assistant_turn_index": assistant_turn,
                                    "negative_kind": "solver_teacher",
                                }
                            )
                            pair_counts[meta["game"]] += 1
                            turn_counts[assistant_turn] += 1
                            negative_kind_counts["solver_teacher"] += 1

                if "repeat_last" in negative_modes and prior_guesses:
                    rejected = replace_guess(current_assistant["content"], prior_guesses[-1])
                    if rejected and rejected != current_assistant["content"]:
                        rejected_variants.append(("repeat_last", rejected))

                if "repeat_first" in negative_modes and prior_guesses:
                    rejected = replace_guess(current_assistant["content"], prior_guesses[0])
                    if rejected and rejected != current_assistant["content"]:
                        rejected_variants.append(("repeat_first", rejected))

                if "bad_length" in negative_modes and len(chosen_guess) >= 5:
                    rejected = replace_guess(current_assistant["content"], chosen_guess[:4])
                    if rejected and rejected != current_assistant["content"]:
                        rejected_variants.append(("bad_length", rejected))

                if "invalid_word" in negative_modes:
                    invalid_guess = make_invalid_word(chosen_guess, valid_words)
                    rejected = replace_guess(current_assistant["content"], invalid_guess)
                    if rejected and rejected != current_assistant["content"]:
                        rejected_variants.append(("invalid_word", rejected))

                if "duplicate_guess_keyword" in negative_modes:
                    rejected = current_assistant["content"].rstrip() + f"\nguess: {chosen_guess}"
                    if rejected != current_assistant["content"]:
                        rejected_variants.append(("duplicate_guess_keyword", rejected))

                if "low_info_valid" in negative_modes:
                    rejected_guess = select_low_info_valid_guess(
                        valid_words=valid_words,
                        state=state,
                        prior_guesses=prior_guesses,
                        chosen_guess=chosen_guess,
                        min_gap=min_negative_gap,
                    )
                    if rejected_guess:
                        rejected = replace_guess(current_assistant["content"], rejected_guess)
                        if rejected and rejected != current_assistant["content"]:
                            rejected_variants.append(("low_info_valid", rejected))

                if "missing_required_valid" in negative_modes:
                    rejected_guess = select_missing_required_valid_guess(
                        valid_words=valid_words,
                        state=state,
                        prior_guesses=prior_guesses,
                        chosen_guess=chosen_guess,
                    )
                    if rejected_guess:
                        rejected = replace_guess(current_assistant["content"], rejected_guess)
                        if rejected and rejected != current_assistant["content"]:
                            rejected_variants.append(("missing_required_valid", rejected))

                if "used_red_valid" in negative_modes:
                    rejected_guess = select_used_red_valid_guess(
                        valid_words=valid_words,
                        state=state,
                        prior_guesses=prior_guesses,
                        chosen_guess=chosen_guess,
                    )
                    if rejected_guess:
                        rejected = replace_guess(current_assistant["content"], rejected_guess)
                        if rejected and rejected != current_assistant["content"]:
                            rejected_variants.append(("used_red_valid", rejected))

                if "bad_yellow_valid" in negative_modes:
                    rejected_guess = select_bad_yellow_valid_guess(
                        valid_words=valid_words,
                        state=state,
                        prior_guesses=prior_guesses,
                        chosen_guess=chosen_guess,
                    )
                    if rejected_guess:
                        rejected = replace_guess(current_assistant["content"], rejected_guess)
                        if rejected and rejected != current_assistant["content"]:
                            rejected_variants.append(("bad_yellow_valid", rejected))

                if "clean_trap" in negative_modes:
                    rejected_guess = select_trap_guess(
                        trap_guesses=trap_guesses,
                        valid_words=valid_words,
                        state=state,
                        prior_guesses=prior_guesses,
                        chosen_guess=chosen_guess,
                    )
                    if rejected_guess:
                        rejected = replace_guess(current_assistant["content"], rejected_guess)
                        if rejected and rejected != current_assistant["content"]:
                            rejected_variants.append(("clean_trap", rejected))

                if "invalid_length_trap" in negative_modes:
                    rejected_guess = select_invalid_length_trap_guess(
                        trap_guesses=long_trap_guesses,
                        prior_guesses=prior_guesses,
                        chosen_guess=chosen_guess,
                    )
                    if rejected_guess:
                        rejected = replace_guess(current_assistant["content"], rejected_guess)
                        if rejected and rejected != current_assistant["content"]:
                            rejected_variants.append(("invalid_length_trap", rejected))

                if sample_one_negative and rejected_variants:
                    rejected_variants = [rng.choice(rejected_variants)]

                for negative_kind, rejected_text in rejected_variants:
                    rows.append(
                        {
                            "prompt": prompt,
                            "chosen": [current_assistant],
                            "rejected": [{"role": "assistant", "content": rejected_text}],
                            "game": meta["game"],
                            "game_role": meta["game_role"],
                            "chosen_model": meta["model"],
                            "assistant_turn_index": assistant_turn,
                            "negative_kind": negative_kind,
                        }
                    )
                    pair_counts[meta["game"]] += 1
                    turn_counts[assistant_turn] += 1
                    negative_kind_counts[negative_kind] += 1

            history.append(current_assistant)
            if chosen_guess:
                prior_guesses.append(chosen_guess)

    rng.shuffle(rows)
    return rows, pair_counts, source_counts, turn_counts, negative_kind_counts


class Qwen35WordleTurnDpoTrainer(BasePlaypenTrainer):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)

    def learn(self):
        dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train",
                                   revision=os.getenv("PLAYPEN_TRAIN_DATASET_REVISION") or None)
        seed = env_int("PLAYPEN_SEED", 42)
        source_models_filter = set(env_csv("PLAYPEN_SOURCE_MODELS"))
        negative_modes = env_csv("PLAYPEN_NEGATIVE_MODES") or ["repeat_last", "bad_length"]
        min_assistant_turn = env_int("PLAYPEN_MIN_ASSISTANT_TURN", 2)
        max_assistant_turn = env_int("PLAYPEN_MAX_ASSISTANT_TURN", 0)
        required_prev_user_substring = env_str("PLAYPEN_REQUIRE_PREV_USER_SUBSTRING", "guess_feedback:")
        clean_only = env_flag("PLAYPEN_WORDLE_CLEAN_ONLY", True)
        max_pairs = env_int("PLAYPEN_MAX_PAIRS", 0)
        sample_one_negative = env_flag("PLAYPEN_SAMPLE_ONE_NEGATIVE", False)
        min_negative_gap = env_float("PLAYPEN_MIN_NEGATIVE_GAP", 0.0)
        trap_guesses = env_csv("PLAYPEN_TRAP_GUESSES") or ["clean"]
        long_trap_guesses = env_csv("PLAYPEN_LONG_TRAP_GUESSES") or [
            "bottle",
            "bridge",
            "teaser",
            "drinker",
            "breeze",
            "hugger",
        ]
        solver_top_k = env_int("PLAYPEN_SOLVER_TOP_K", 4)
        solver_max_exact_pool = env_int("PLAYPEN_SOLVER_MAX_EXACT_POOL", 256)
        min_solver_gain = env_float("PLAYPEN_MIN_SOLVER_GAIN", 0.0)
        valid_words_path = env_str("PLAYPEN_WORDLE_VALID_WORDS_FILE", default_valid_words_file())
        valid_words = load_valid_words(valid_words_path)

        preference_rows, pair_counts, source_counts, turn_counts, negative_kind_counts = build_wordle_turn_preferences(
            dataset=dataset,
            seed=seed,
            source_models_filter=source_models_filter,
            negative_modes=negative_modes,
            min_assistant_turn=min_assistant_turn,
            max_assistant_turn=max_assistant_turn,
            required_prev_user_substring=required_prev_user_substring,
            clean_only=clean_only,
            valid_words=valid_words,
            sample_one_negative=sample_one_negative,
            min_negative_gap=min_negative_gap,
            trap_guesses=trap_guesses,
            long_trap_guesses=long_trap_guesses,
            solver_top_k=solver_top_k,
            solver_max_exact_pool=solver_max_exact_pool,
            min_solver_gain=min_solver_gain,
        )

        if not preference_rows:
            raise ValueError("No wordle turn-level preference pairs were created for the requested configuration.")

        if max_pairs > 0:
            preference_rows = preference_rows[:max_pairs]

        print("Preference pairs:", len(preference_rows))
        print("Pairs by game:", pair_counts.most_common())
        print("Source models:", source_counts.most_common(10))
        print("Assistant turn indices:", sorted(turn_counts.items()))
        print("Negative kinds:", negative_kind_counts.most_common())
        print("Max assistant turn:", max_assistant_turn)
        print("Sample one negative:", sample_one_negative)
        print("Min negative gap:", min_negative_gap)
        print("Solver top-k:", solver_top_k)
        print("Solver max exact pool:", solver_max_exact_pool)
        print("Min solver gain:", min_solver_gain)
        print("Trap guesses:", trap_guesses)
        print("Long trap guesses:", long_trap_guesses)
        print("Valid words loaded:", len(valid_words))
        print("Example pair prompt:")
        for message in preference_rows[0]["prompt"][-4:]:
            print(message)
        print("Example chosen:")
        for message in preference_rows[0]["chosen"]:
            print(message)
        print("Example rejected:")
        for message in preference_rows[0]["rejected"]:
            print(message)

        preference_dataset = Dataset.from_list(preference_rows)
        eval_ratio = env_float("PLAYPEN_EVAL_RATIO", 0.1)
        if len(preference_dataset) < 2:
            print("Preference dataset too small for train/eval split; disabling eval.")
            preference_dataset = {"train": preference_dataset, "test": None}
        else:
            preference_dataset = preference_dataset.train_test_split(test_size=eval_ratio, shuffle=True, seed=seed)

        output_dir = os.getenv("PLAYPEN_OUTPUT_DIR", f"models/dpo+lora/{self.learner.name}")
        max_length = env_int("PLAYPEN_MAX_LENGTH", 1024)
        batch_size = env_int("PLAYPEN_BATCH_SIZE", 1)
        grad_accum = env_int("PLAYPEN_GRAD_ACCUM", 16)
        gradient_checkpointing = env_flag("PLAYPEN_GRADIENT_CHECKPOINTING", False)
        learning_rate = env_float("PLAYPEN_LR", 5e-6)
        num_epochs = env_float("PLAYPEN_NUM_TRAIN_EPOCHS", 1.0)
        save_steps = env_int("PLAYPEN_SAVE_STEPS", 100)
        eval_steps = env_int("PLAYPEN_EVAL_STEPS", 100)
        logging_steps = env_int("PLAYPEN_LOGGING_STEPS", 10)
        save_total_limit = env_int("PLAYPEN_SAVE_TOTAL_LIMIT", 2)
        beta = env_float("PLAYPEN_DPO_BETA", 0.1)
        use_separate_ref_model = env_flag("PLAYPEN_USE_SEPARATE_REF_MODEL", True)
        has_eval_dataset = preference_dataset["test"] is not None

        self.learner.model.config.use_cache = not gradient_checkpointing
        is_existing_peft = isinstance(self.learner.model, PeftModel)
        if is_existing_peft:
            print("Continuing DPO training from an existing PEFT adapter.")
            adapter_name = getattr(self.learner.model, "active_adapter", "default")
            if isinstance(adapter_name, list):
                adapter_name = adapter_name[0]
            self.learner.model.set_adapter(adapter_name)
            self.learner.model.train()
            self.learner.model.print_trainable_parameters()
        print("Gradient checkpointing:", gradient_checkpointing)
        print("Use separate ref model:", use_separate_ref_model)

        ref_model = None
        if is_existing_peft and use_separate_ref_model:
            ref_model = copy.deepcopy(self.learner.model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False

        config = trl.DPOConfig(
            output_dir=output_dir,
            max_length=max_length,
            eval_strategy="steps" if has_eval_dataset else "no",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
            bf16=True,
            report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
            beta=beta,
            truncation_mode="keep_end",
        )

        trainer = trl.DPOTrainer(
            model=self.learner.model,
            ref_model=ref_model,
            args=config,
            train_dataset=preference_dataset["train"],
            eval_dataset=preference_dataset["test"],
            processing_class=self.learner.tokenizer,
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
