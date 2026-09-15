import copy
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import trl
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel

from clemcore.backends import Model
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import (
    EpochResultsFolder,
    EpochResultsFolderCallback,
    ExperimentFileSaver,
    GameBenchmark,
    GameBenchmarkCallbackList,
    GameInstances,
    GameRegistry,
    InstanceFileSaver,
    InteractionsFileSaver,
)
from clemcore.clemgame.runners import branching

from playpen import BasePlaypenTrainer, to_instances_filter
from playpen.buffers import BranchingEpisodeBuffer
from playpen.callbacks.buffers import BranchingEpisodeBufferCallback

TRL_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(TRL_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(TRL_EXAMPLES_DIR))

from wordle_solver import top_k_guesses


WORDLE_GUESS_FIELD_RE = re.compile(r"^guess:\s*(.*?)\s*$", re.MULTILINE | re.IGNORECASE)
WORDLE_FEEDBACK_LINE_RE = re.compile(r"^\s*guess_feedback:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


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


def default_valid_words_file() -> str:
    return "clembench/wordle/resources/target_words/en/official_recognized_words.txt"


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


def extract_guess_fields(text: str) -> list[str]:
    return [match.group(1).strip().lower() for match in WORDLE_GUESS_FIELD_RE.finditer(text)]


def extract_guess_feedback(text: str) -> str | None:
    match = WORDLE_FEEDBACK_LINE_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def analyze_wordle_reply(text: str, valid_words: set[str]) -> dict[str, int]:
    guess_fields = extract_guess_fields(text)
    bad_length = 0
    invalid_word = 0
    non_alpha = 0

    for guess in guess_fields:
        if not guess.isalpha():
            non_alpha += 1
            continue
        if len(guess) != 5:
            bad_length += 1
            continue
        if valid_words and guess not in valid_words:
            invalid_word += 1

    has_issue = int(
        len(guess_fields) == 0
        or len(guess_fields) > 1
        or bad_length > 0
        or invalid_word > 0
        or non_alpha > 0
    )
    return {
        "guess_count": len(guess_fields),
        "missing_guess": int(len(guess_fields) == 0),
        "duplicate_guess_keyword": int(len(guess_fields) > 1),
        "bad_length": bad_length,
        "invalid_word": invalid_word,
        "non_alpha": non_alpha,
        "is_valid": int(has_issue == 0),
    }


def summarize_pair_reply_validity(preference_rows: list[dict], valid_words: set[str]) -> dict[str, int]:
    counts = Counter()
    for row in preference_rows:
        for side in ("chosen", "rejected"):
            content = row[side][0]["content"]
            info = analyze_wordle_reply(content, valid_words)
            counts[f"{side}_valid"] += info["is_valid"]
            counts[f"{side}_missing_guess"] += info["missing_guess"]
            counts[f"{side}_duplicate_guess_keyword"] += info["duplicate_guess_keyword"]
            counts[f"{side}_bad_length"] += info["bad_length"]
            counts[f"{side}_invalid_word"] += info["invalid_word"]
            counts[f"{side}_non_alpha"] += info["non_alpha"]
    return dict(sorted(counts.items()))


def count_identical_reply_pairs(preference_rows: list[dict]) -> int:
    return sum(
        1
        for row in preference_rows
        if row["chosen"][0]["content"].strip() == row["rejected"][0]["content"].strip()
    )


def build_branch_preference_rows(
    branching_points: list,
    player_name: str,
    valid_words: set[str],
    *,
    require_different_scores: bool = True,
    require_valid_chosen: bool = False,
    require_valid_rejected: bool = False,
    require_distinct_reply: bool = False,
    min_score_gap: float = 0.0,
) -> tuple[list[dict], dict[str, float | int]]:
    groups = defaultdict(list)
    for branching_point in branching_points:
        groups[branching_point.game_snapshot.origin].append(branching_point)

    rows: list[dict] = []
    score_gaps: list[float] = []
    stats = Counter()

    for siblings in groups.values():
        stats["groups_total"] += 1
        if len(siblings) < 2:
            stats["groups_too_small"] += 1
            continue

        ranked = sorted(siblings, key=lambda s: s.episode_score, reverse=True)

        def reply_is_valid(branch_point) -> bool:
            return bool(analyze_wordle_reply(branch_point.diverging_step.response, valid_words)["is_valid"])

        chosen_candidates = ranked
        if require_valid_chosen:
            chosen_candidates = [s for s in chosen_candidates if reply_is_valid(s)]
            if not chosen_candidates:
                stats["groups_no_valid_chosen"] += 1
                continue

        chosen = chosen_candidates[0]
        rejected_candidates = list(reversed(ranked))
        if require_valid_rejected:
            rejected_candidates = [s for s in rejected_candidates if reply_is_valid(s)]
            if not rejected_candidates:
                stats["groups_no_valid_rejected"] += 1
                continue

        rejected = None
        chosen_text = chosen.diverging_step.response.strip()
        for candidate in rejected_candidates:
            if candidate is chosen:
                continue
            if require_distinct_reply and candidate.diverging_step.response.strip() == chosen_text:
                continue
            rejected = candidate
            break

        if rejected is None:
            stats["groups_no_rejected_after_filters"] += 1
            continue

        score_gap = float(chosen.episode_score - rejected.episode_score)
        if require_different_scores and math.isclose(score_gap, 0.0, abs_tol=1e-6):
            stats["groups_same_score"] += 1
            continue
        if score_gap < min_score_gap:
            stats["groups_below_score_gap"] += 1
            continue

        prompt = []
        for step in chosen.prompt:
            if step.player_name == player_name:
                prompt.append(step.context)
                prompt.append({"role": "assistant", "content": step.response})
        prompt.append(chosen.diverging_step.context)

        rows.append(
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen.diverging_step.response}],
                "rejected": [{"role": "assistant", "content": rejected.diverging_step.response}],
            }
        )
        score_gaps.append(score_gap)

    summary: dict[str, float | int] = dict(sorted(stats.items()))
    summary["groups_with_pairs"] = len(rows)
    if score_gaps:
        ordered = sorted(score_gaps)
        summary["score_gap_min"] = ordered[0]
        summary["score_gap_p50"] = ordered[len(ordered) // 2]
        summary["score_gap_max"] = ordered[-1]
        summary["score_gap_mean"] = sum(ordered) / len(ordered)
    else:
        summary["score_gap_min"] = 0.0
        summary["score_gap_p50"] = 0.0
        summary["score_gap_max"] = 0.0
        summary["score_gap_mean"] = 0.0
    return rows, summary


def extract_words_by_color_code(guess_word: str):
    color_label_dict = {}
    letters_list = []

    for letter_code in guess_word.split(" "):
        matches = re.findall(r"\b(\w+)\b\s*\<(.+?)\>", letter_code)
        for match in matches:
            letter = match[0].strip()
            letters_list.append(letter)
            color_code = match[1].strip()
            color_label_dict.setdefault(color_code, []).append(letter)
    return color_label_dict, letters_list


@dataclass
class WordleConstraintState:
    required_letters: set[str] = field(default_factory=set)
    banned_letters: set[str] = field(default_factory=set)
    fixed_positions: dict[int, str] = field(default_factory=dict)
    banned_positions: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))


def parse_feedback_tokens(feedback: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for token in feedback.split():
        matches = re.findall(r"\b(\w+)\b\s*\<(.+?)\>", token)
        for letter, color in matches:
            pairs.append((letter.lower(), color.lower()))
    return pairs


def update_constraint_state(state: WordleConstraintState, guess: str, feedback: str) -> None:
    parsed = parse_feedback_tokens(feedback)
    if not parsed or len(parsed) != len(guess):
        return

    present_letters = {letter for letter, color in parsed if color in {"green", "yellow"}}
    for idx, (letter, color) in enumerate(parsed):
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


def classify_constraint_violations(state: WordleConstraintState, guess: str) -> Counter:
    counts = Counter()
    if len(guess) != 5 or not guess.isalpha():
        return counts

    for idx, letter in state.fixed_positions.items():
        if guess[idx] != letter:
            counts["missed_green"] += 1
            break

    for idx, letter in enumerate(guess):
        if letter in state.banned_letters:
            counts["used_red_letter"] += 1
            break
        if idx in state.banned_positions.get(letter, set()):
            counts["bad_yellow_position"] += 1
            break

    if any(letter not in guess for letter in state.required_letters):
        counts["missing_required_letter"] += 1

    return counts


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


def infer_state_from_messages(messages: list[dict]) -> tuple[WordleConstraintState, list[str], int]:
    state = WordleConstraintState()
    prior_guesses: list[str] = []
    assistant_turn = 0

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant":
            assistant_turn += 1
            guess_fields = extract_guess_fields(content)
            if guess_fields:
                prior_guesses.append(guess_fields[-1])
        elif role == "user" and prior_guesses:
            feedback = extract_guess_feedback(content)
            if feedback:
                update_constraint_state(state, prior_guesses[-1], feedback)

    return state, prior_guesses, assistant_turn + 1


def remaining_valid_words(valid_words: set[str], state: WordleConstraintState, prior_guesses: list[str]) -> list[str]:
    blocked = set(prior_guesses)
    return [word for word in valid_words if word not in blocked and word_matches_state(word, state)]


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


def score_wordle_reply(
    text: str,
    valid_words: set[str],
    state: WordleConstraintState,
    prior_guesses: list[str],
    quality_weight: float,
) -> float:
    info = analyze_wordle_reply(text, valid_words)
    score = 0.0
    score -= 500.0 * info["missing_guess"]
    score -= 450.0 * info["duplicate_guess_keyword"]
    score -= 450.0 * info["bad_length"]
    score -= 450.0 * info["invalid_word"]
    score -= 450.0 * info["non_alpha"]

    guess_fields = extract_guess_fields(text)
    if len(guess_fields) != 1:
        return score

    guess = guess_fields[-1]
    if guess in prior_guesses:
        score -= 200.0

    violations = classify_constraint_violations(state, guess)
    score -= 180.0 * violations["missed_green"]
    score -= 140.0 * violations["bad_yellow_position"]
    score -= 180.0 * violations["missing_required_letter"]
    score -= 140.0 * violations["used_red_letter"]

    candidate_pool = remaining_valid_words(valid_words, state, prior_guesses)
    if guess not in candidate_pool and len(guess) == 5 and guess.isalpha():
        candidate_pool = candidate_pool + [guess]
    quality_scores = candidate_quality_scores(candidate_pool)
    score += quality_weight * quality_scores.get(guess, 0.0)
    if info["is_valid"]:
        score += 1000.0
    return score


@contextmanager
def temporary_valid_reply_retry(
    learner,
    *,
    valid_words: set[str],
    retries: int,
    retry_temperature_step: float,
    min_turn: int,
    max_turn: int,
    quality_weight: float,
    solver_top_k: int,
    solver_max_exact_pool: int,
):
    if retries <= 0 and solver_top_k <= 0:
        yield Counter()
        return

    original_generate_response = learner.__class__.generate_response
    stats = Counter()
    target_name = learner.name
    solver_prompt_counts = Counter()

    def make_solver_response(word: str) -> str:
        return (
            "explanation: I am choosing a valid candidate that matches the current feedback "
            "and reduces the remaining possibilities.\n"
            f"guess: {word}"
        )

    def wrapped(model_self, messages):
        if model_self.name != target_name:
            return original_generate_response(model_self, messages)

        state, prior_guesses, next_turn = infer_state_from_messages(messages)
        last_user = messages[-1]["content"] if messages and messages[-1].get("role") == "user" else ""
        if next_turn < min_turn:
            return original_generate_response(model_self, messages)
        if max_turn >= 0 and next_turn > max_turn:
            return original_generate_response(model_self, messages)
        if "guess_feedback:" not in last_user.lower():
            return original_generate_response(model_self, messages)

        candidate_pool = remaining_valid_words(valid_words, state, prior_guesses)
        if solver_top_k > 0 and candidate_pool:
            key = tuple((message.get("role"), message.get("content", "")) for message in messages)
            top_candidates = top_k_guesses(candidate_pool, solver_top_k, max_exact_pool=solver_max_exact_pool)
            if top_candidates:
                selected_index = solver_prompt_counts[key] % len(top_candidates)
                solver_prompt_counts[key] += 1
                selected_word = top_candidates[selected_index]
                base_prompt, base_response, _ = original_generate_response(model_self, messages)
                response_text = make_solver_response(selected_word)
                info = analyze_wordle_reply(response_text, valid_words)
                stats["solver_calls"] += 1
                stats["solver_candidates_available"] += len(top_candidates)
                stats["solver_selected_valid"] += info["is_valid"]
                stats["solver_selected_invalid"] += int(info["is_valid"] == 0)
                return base_prompt, base_response, response_text

        base_gen_args = dict(model_self.gen_args)
        base_temperature = float(base_gen_args.get("temperature", 0.0) or 0.0)
        attempts: list[tuple[float, tuple]] = []
        try:
            for attempt_idx in range(retries + 1):
                attempt_args = dict(base_gen_args)
                if attempt_idx > 0:
                    attempt_args["temperature"] = base_temperature + retry_temperature_step * attempt_idx
                model_self.set_gen_args(**attempt_args)
                result = original_generate_response(model_self, messages)
                score = score_wordle_reply(
                    result[2],
                    valid_words=valid_words,
                    state=state,
                    prior_guesses=prior_guesses,
                    quality_weight=quality_weight,
                )
                attempts.append((score, result))
                stats["attempts_total"] += 1

            best_score, best_result = max(attempts, key=lambda item: item[0])
            best_info = analyze_wordle_reply(best_result[2], valid_words)
            stats["retry_calls"] += 1
            stats["selected_valid"] += best_info["is_valid"]
            stats["selected_invalid"] += int(best_info["is_valid"] == 0)
            stats["selected_score_positive"] += int(best_score > 0.0)
            return best_result
        finally:
            model_self.set_gen_args(**base_gen_args)

    learner.__class__.generate_response = wrapped
    try:
        yield stats
    finally:
        learner.__class__.generate_response = original_generate_response


def information_gain_score(valid_words: set[str], guesses: list[str], feedbacks: list[str]) -> float:
    if not valid_words:
        return 0.0
    state = WordleConstraintState()
    candidates = {word for word in valid_words if word_matches_state(word, state)}
    total_reduction = 0.0
    for guess, feedback in zip(guesses, feedbacks):
        before = max(len(candidates), 1)
        update_constraint_state(state, guess, feedback)
        candidates = {word for word in candidates if word_matches_state(word, state)}
        total_reduction += 1.0 - (len(candidates) / before)
    return total_reduction


def turns_closeness(guesser_feedbacks: list[str]) -> list[int]:
    score_list = []
    for feedback in guesser_feedbacks:
        score = 0
        for letter in feedback.split(" "):
            if "green" in letter:
                score += 5
            elif "yellow" in letter:
                score += 3
        score_list.append(score)
    return score_list


def turns_strategy(guesser_feedbacks: list[str], is_aborted: bool) -> list[int]:
    if len(guesser_feedbacks) == 1:
        if is_aborted:
            return [0]
        return [100]

    score_list = [0]
    for guess1, guess2 in zip(guesser_feedbacks, guesser_feedbacks[1:]):
        guess1_dict, _ = extract_words_by_color_code(guess1)
        _, guess2_letters = extract_words_by_color_code(guess2)
        guess1_not_use = guess1_dict.get("red", [])
        guess1_use = guess1_dict.get("green", [])
        guess1_change = guess1_dict.get("yellow", [])
        score = 0

        result = len(set(guess1_not_use) & set(guess2_letters))
        if result:
            score -= result * 20

        result = len(set(guess1_use) & set(guess2_letters))
        if result:
            score += result * 20

        result = len(set(guess1_change) & set(guess2_letters))
        if result:
            score += result * 10

        score_list.append(score)
    return score_list


SPEED_SCORES = {
    1: 100,
    2: 100,
    3: 100,
    4: 50,
    5: 30,
    6: 20,
}


def mean_terminal_reward(rewards) -> float:
    if isinstance(rewards, dict) and rewards:
        return float(sum(rewards.values()) / len(rewards))
    try:
        return float(rewards)
    except (TypeError, ValueError):
        return 0.0


class WordleBranchingEpisodeBufferCallback(BranchingEpisodeBufferCallback):
    def __init__(
        self,
        episode_buffer: BranchingEpisodeBuffer,
        *,
        success_base: float = 1000.0,
        failure_base: float = 100.0,
        abort_base: float = -500.0,
        speed_weight: float = 2.0,
        closeness_weight: float = 3.0,
        strategy_weight: float = 0.25,
        repeat_penalty: float = 80.0,
        violated_penalty: float = 60.0,
        unparsed_penalty: float = 5.0,
        invalid_word_penalty: float = 120.0,
        bad_length_penalty: float = 120.0,
        duplicate_guess_penalty: float = 80.0,
        missing_guess_penalty: float = 80.0,
        non_alpha_penalty: float = 80.0,
        missed_green_penalty: float = 60.0,
        bad_yellow_position_penalty: float = 50.0,
        missing_required_letter_penalty: float = 60.0,
        used_red_letter_penalty: float = 50.0,
        info_gain_weight: float = 0.0,
        valid_words: set[str] | None = None,
    ):
        super().__init__(episode_buffer)
        self.success_base = success_base
        self.failure_base = failure_base
        self.abort_base = abort_base
        self.speed_weight = speed_weight
        self.closeness_weight = closeness_weight
        self.strategy_weight = strategy_weight
        self.repeat_penalty = repeat_penalty
        self.violated_penalty = violated_penalty
        self.unparsed_penalty = unparsed_penalty
        self.invalid_word_penalty = invalid_word_penalty
        self.bad_length_penalty = bad_length_penalty
        self.duplicate_guess_penalty = duplicate_guess_penalty
        self.missing_guess_penalty = missing_guess_penalty
        self.non_alpha_penalty = non_alpha_penalty
        self.missed_green_penalty = missed_green_penalty
        self.bad_yellow_position_penalty = bad_yellow_position_penalty
        self.missing_required_letter_penalty = missing_required_letter_penalty
        self.used_red_letter_penalty = used_red_letter_penalty
        self.info_gain_weight = info_gain_weight
        self.valid_words = valid_words or set()

    def __deepcopy__(self, memo):
        new = WordleBranchingEpisodeBufferCallback(
            self.episode_buffer,
            success_base=self.success_base,
            failure_base=self.failure_base,
            abort_base=self.abort_base,
            speed_weight=self.speed_weight,
            closeness_weight=self.closeness_weight,
            strategy_weight=self.strategy_weight,
            repeat_penalty=self.repeat_penalty,
            violated_penalty=self.violated_penalty,
            unparsed_penalty=self.unparsed_penalty,
            invalid_word_penalty=self.invalid_word_penalty,
            bad_length_penalty=self.bad_length_penalty,
            duplicate_guess_penalty=self.duplicate_guess_penalty,
            missing_guess_penalty=self.missing_guess_penalty,
            non_alpha_penalty=self.non_alpha_penalty,
            missed_green_penalty=self.missed_green_penalty,
            bad_yellow_position_penalty=self.bad_yellow_position_penalty,
            missing_required_letter_penalty=self.missing_required_letter_penalty,
            used_red_letter_penalty=self.used_red_letter_penalty,
            info_gain_weight=self.info_gain_weight,
            valid_words=self.valid_words,
        )
        new._trajectory_markers = copy.deepcopy(self._trajectory_markers, memo)
        new._trajectory = copy.deepcopy(self._trajectory, memo)
        return new

    def _episode_score(self, game_master, rewards, trajectory=None) -> float:
        terminal_reward = mean_terminal_reward(rewards)
        guesses = list(getattr(game_master, "guesser_guesses", []))
        feedbacks = list(getattr(game_master, "guesser_feedbacks", []))
        request_count = int(getattr(game_master, "request_counts", 0) or 0)
        parsed_count = int(getattr(game_master, "parsed_request_counts", 0) or 0)
        violated_count = int(getattr(game_master, "violated_request_counts", 0) or 0)
        missing_parsed = max(request_count - parsed_count, 0)
        guess_repetitions = len(guesses) - len(set(guesses))
        closeness_scores = turns_closeness(feedbacks) if feedbacks else []
        strategy_scores = turns_strategy(feedbacks, is_aborted=terminal_reward < -0.5) if feedbacks else []
        reply_issue_counts = Counter()
        constraint_issue_counts = Counter()
        info_gain = information_gain_score(self.valid_words, guesses, feedbacks)

        for step in trajectory or []:
            response = getattr(step, "response", "")
            if not response:
                continue
            info = analyze_wordle_reply(response, self.valid_words)
            reply_issue_counts["missing_guess"] += info["missing_guess"]
            reply_issue_counts["duplicate_guess_keyword"] += info["duplicate_guess_keyword"]
            reply_issue_counts["bad_length"] += info["bad_length"]
            reply_issue_counts["invalid_word"] += info["invalid_word"]
            reply_issue_counts["non_alpha"] += info["non_alpha"]

        state = WordleConstraintState()
        for guess, feedback in zip(guesses, feedbacks):
            constraint_issue_counts.update(classify_constraint_violations(state, guess))
            update_constraint_state(state, guess, feedback)

        if terminal_reward > 0.5:
            score = self.success_base + self.speed_weight * SPEED_SCORES.get(len(guesses), 0)
        elif terminal_reward < -0.5:
            score = self.abort_base
        else:
            score = self.failure_base

        score += self.closeness_weight * sum(closeness_scores)
        score += self.strategy_weight * sum(strategy_scores)
        score -= self.repeat_penalty * guess_repetitions
        score -= self.violated_penalty * violated_count
        score -= self.unparsed_penalty * missing_parsed
        score -= self.invalid_word_penalty * reply_issue_counts["invalid_word"]
        score -= self.bad_length_penalty * reply_issue_counts["bad_length"]
        score -= self.duplicate_guess_penalty * reply_issue_counts["duplicate_guess_keyword"]
        score -= self.missing_guess_penalty * reply_issue_counts["missing_guess"]
        score -= self.non_alpha_penalty * reply_issue_counts["non_alpha"]
        score -= self.missed_green_penalty * constraint_issue_counts["missed_green"]
        score -= self.bad_yellow_position_penalty * constraint_issue_counts["bad_yellow_position"]
        score -= self.missing_required_letter_penalty * constraint_issue_counts["missing_required_letter"]
        score -= self.used_red_letter_penalty * constraint_issue_counts["used_red_letter"]
        score += self.info_gain_weight * info_gain
        return float(score)

    def on_game_end(self, game_master, game_instance, exception=None, rewards=None):
        if exception is not None:
            return
        episode_score = self._episode_score(game_master, rewards, self._trajectory)
        for game_snapshot, turn in self._trajectory_markers:
            self.episode_buffer.add_branching_point(game_snapshot, turn, self._trajectory, episode_score)
        self._trajectory = []
        self._trajectory_markers = []


def make_round_window_condition(min_round: int, max_round: int):
    def condition(player=None, env=None, **_):
        gm = env.game_master if env is not None else None
        if gm is None or not hasattr(gm, "current_round"):
            return False
        if gm.current_round < min_round:
            return False
        if max_round >= 0 and gm.current_round > max_round:
            return False
        return True

    return condition


class Qwen35WordleBranchDpoTrainer(BasePlaypenTrainer):
    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        self.episode_buffer = BranchingEpisodeBuffer()

    def learn(self):
        seed = env_int("PLAYPEN_SEED", 42)
        player_name = env_str("PLAYPEN_PLAYER_NAME", "Player 1")
        branching_factor = env_int("PLAYPEN_BRANCHING_FACTOR", 2)
        min_round = env_int("PLAYPEN_MIN_ROUND", 1)
        max_round = env_int("PLAYPEN_MAX_ROUND", 3)
        max_instances = env_int("PLAYPEN_MAX_INSTANCES", 0)
        branch_temperature = env_float("PLAYPEN_TEMPERATURE", 0.7)
        branch_max_tokens = env_int("PLAYPEN_MAX_TOKENS", 256)
        max_pairs = env_int("PLAYPEN_MAX_PAIRS", 0)
        valid_words_file = env_str("PLAYPEN_VALID_WORDS_FILE", default_valid_words_file())
        require_valid_chosen = env_flag("PLAYPEN_REQUIRE_VALID_CHOSEN", False)
        require_valid_rejected = env_flag("PLAYPEN_REQUIRE_VALID_REJECTED", False)
        require_distinct_reply = env_flag("PLAYPEN_REQUIRE_DISTINCT_REPLY", False)
        min_score_gap = env_float("PLAYPEN_MIN_SCORE_GAP", 0.0)
        valid_reply_retries = env_int("PLAYPEN_VALID_REPLY_RETRIES", 0)
        valid_reply_temp_step = env_float("PLAYPEN_VALID_REPLY_TEMP_STEP", 0.15)
        valid_reply_min_turn = env_int("PLAYPEN_VALID_REPLY_MIN_TURN", min_round)
        valid_reply_max_turn = env_int("PLAYPEN_VALID_REPLY_MAX_TURN", max_round)
        reply_quality_weight = env_float("PLAYPEN_REPLY_QUALITY_WEIGHT", 40.0)
        solver_top_k = env_int("PLAYPEN_SOLVER_TOP_K", 0)
        solver_max_exact_pool = env_int("PLAYPEN_SOLVER_MAX_EXACT_POOL", 256)
        valid_words = load_valid_words(valid_words_file)
        dump_pairs_path = env_str("PLAYPEN_DUMP_BRANCH_PAIRS", "")
        save_pairs_path = env_str("PLAYPEN_SAVE_BRANCH_PAIRS", "")
        load_pairs_path = env_str("PLAYPEN_BRANCH_PREF_DATA", "")
        if sum(bool(path) for path in (dump_pairs_path, save_pairs_path, load_pairs_path)) > 1:
            raise ValueError("Choose only one of PLAYPEN_DUMP_BRANCH_PAIRS, PLAYPEN_SAVE_BRANCH_PAIRS, and PLAYPEN_BRANCH_PREF_DATA")

        self.learner.set_gen_args(max_tokens=branch_max_tokens, temperature=branch_temperature)
        print("Branch generation temperature:", branch_temperature)
        print("Branch generation max_tokens:", branch_max_tokens)
        print("Loaded valid Wordle words:", len(valid_words))
        print("Valid reply retries:", valid_reply_retries)
        print("Valid reply retry temperature step:", valid_reply_temp_step)
        print("Valid reply retry turn window:", (valid_reply_min_turn, valid_reply_max_turn))
        print("Reply quality weight:", reply_quality_weight)
        print("Solver top-k:", solver_top_k)
        print("Solver max exact pool:", solver_max_exact_pool)

        if load_pairs_path:
            import gzip as _gzip
            load_path = Path(load_pairs_path)
            print(f"Loading pre-built preference pairs from {load_path}")
            rows = []
            with _gzip.open(load_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if not rows:
                raise ValueError(f"No preference pairs found in {load_path}")
            preference_dataset = Dataset.from_list(rows)
            if max_pairs > 0:
                preference_dataset = preference_dataset.shuffle(seed=seed).select(
                    range(min(max_pairs, len(preference_dataset)))
                )
            print(f"Loaded {len(preference_dataset)} preference pairs")
        else:

            dataset = load_dataset("colab-potsdam/playpen-data", "instances", split="train",
                                   revision=os.getenv("PLAYPEN_TRAIN_DATASET_REVISION") or None)
            dataset = dataset.filter(lambda row: row["game"] == "wordle")
            if max_instances > 0:
                dataset = dataset.shuffle(seed=seed).select(range(min(max_instances, len(dataset))))

            experiment_counts = Counter(dataset["experiment"])
            print("Training instances:", len(dataset))
            print("Experiments:", experiment_counts.most_common())

            records_dir = Path(env_str("PLAYPEN_BRANCH_RECORDS_DIR", "playpen-records-branching-wordle"))
            results_folder = EpochResultsFolder(records_dir, Model.to_identifier([self.learner]))
            model_infos = Model.to_infos([self.learner])
            callbacks = GameBenchmarkCallbackList(
                [
                    WordleBranchingEpisodeBufferCallback(
                        self.episode_buffer,
                        success_base=env_float("PLAYPEN_SUCCESS_BASE", 1000.0),
                        failure_base=env_float("PLAYPEN_FAILURE_BASE", 100.0),
                        abort_base=env_float("PLAYPEN_ABORT_BASE", -500.0),
                        speed_weight=env_float("PLAYPEN_SPEED_WEIGHT", 2.0),
                        closeness_weight=env_float("PLAYPEN_CLOSENESS_WEIGHT", 3.0),
                        strategy_weight=env_float("PLAYPEN_STRATEGY_WEIGHT", 0.25),
                        repeat_penalty=env_float("PLAYPEN_REPEAT_PENALTY", 80.0),
                        violated_penalty=env_float("PLAYPEN_VIOLATED_PENALTY", 60.0),
                        unparsed_penalty=env_float("PLAYPEN_UNPARSED_PENALTY", 5.0),
                        invalid_word_penalty=env_float("PLAYPEN_INVALID_WORD_PENALTY", 120.0),
                        bad_length_penalty=env_float("PLAYPEN_BAD_LENGTH_PENALTY", 120.0),
                        duplicate_guess_penalty=env_float("PLAYPEN_DUPLICATE_GUESS_PENALTY", 80.0),
                        missing_guess_penalty=env_float("PLAYPEN_MISSING_GUESS_PENALTY", 80.0),
                        non_alpha_penalty=env_float("PLAYPEN_NON_ALPHA_PENALTY", 80.0),
                        missed_green_penalty=env_float("PLAYPEN_MISSED_GREEN_PENALTY", 60.0),
                        bad_yellow_position_penalty=env_float("PLAYPEN_BAD_YELLOW_POSITION_PENALTY", 50.0),
                        missing_required_letter_penalty=env_float("PLAYPEN_MISSING_REQUIRED_LETTER_PENALTY", 60.0),
                        used_red_letter_penalty=env_float("PLAYPEN_USED_RED_LETTER_PENALTY", 50.0),
                        info_gain_weight=env_float("PLAYPEN_INFO_GAIN_WEIGHT", 0.0),
                        valid_words=valid_words,
                    ),
                    EpochResultsFolderCallback(results_folder),
                    InstanceFileSaver(results_folder),
                    ExperimentFileSaver(results_folder, player_model_infos=model_infos),
                    InteractionsFileSaver(results_folder, player_model_infos=model_infos, store_branches=True),
                ]
            )

            game_registry = GameRegistry.from_directories_and_cwd_files()
            game_spec = game_registry.get_game_specs_that_unify_with("wordle")[0]
            branching_condition = branching.combined_condition(
                branching.is_player_model(self.learner),
                make_round_window_condition(min_round, max_round),
            )

            with GameBenchmark.load_from_spec(game_spec) as game_benchmark:
                game_instances = GameInstances.from_game_spec(game_benchmark.game_spec)
                game_instances = game_instances.filter(to_instances_filter(dataset))
                self.episode_buffer.reset()
                with temporary_valid_reply_retry(
                    self.learner,
                    valid_words=valid_words,
                    retries=valid_reply_retries,
                    retry_temperature_step=valid_reply_temp_step,
                    min_turn=valid_reply_min_turn,
                    max_turn=valid_reply_max_turn,
                    quality_weight=reply_quality_weight,
                    solver_top_k=solver_top_k,
                    solver_max_exact_pool=solver_max_exact_pool,
                ) as retry_stats:
                    branching.run(
                        game_benchmark,
                        game_instances,
                        [self.learner],
                        callbacks=callbacks,
                        branching_factor=branching_factor,
                        branching_condition=branching_condition,
                    )

            raw_preference_rows, raw_pair_selection_summary = build_branch_preference_rows(
                self.episode_buffer._branching_points,
                player_name,
                valid_words,
                require_different_scores=True,
            )
            if len(raw_preference_rows) == 0:
                raise ValueError("No branching preference pairs were created for the requested configuration.")

            raw_pair_validity_summary = summarize_pair_reply_validity(raw_preference_rows, valid_words)
            print("Raw pair reply validity summary:", raw_pair_validity_summary)
            print("Raw pair selection summary:", raw_pair_selection_summary)
            raw_identical_pairs = count_identical_reply_pairs(raw_preference_rows)
            print("Raw identical chosen/rejected reply pairs:", raw_identical_pairs)

            preference_rows, filtered_pair_selection_summary = build_branch_preference_rows(
                self.episode_buffer._branching_points,
                player_name,
                valid_words,
                require_different_scores=True,
                require_valid_chosen=require_valid_chosen,
                require_valid_rejected=require_valid_rejected,
                require_distinct_reply=require_distinct_reply,
                min_score_gap=min_score_gap,
            )
            if not preference_rows:
                raise ValueError("No branching preference pairs remain after applying pair-quality filters.")

            pair_validity_summary = summarize_pair_reply_validity(preference_rows, valid_words)
            print("Final pair reply validity summary:", pair_validity_summary)
            print("Final pair selection summary:", filtered_pair_selection_summary)
            final_identical_pairs = count_identical_reply_pairs(preference_rows)
            print("Final identical chosen/rejected reply pairs:", final_identical_pairs)
            preference_dataset = Dataset.from_list(preference_rows)

            if max_pairs > 0:
                preference_dataset = preference_dataset.shuffle(seed=seed).select(range(min(max_pairs, len(preference_dataset))))

            prompt_message_counts = Counter(len(row["prompt"]) for row in preference_dataset)
            print("Collected full trajectories:", len(self.episode_buffer._trajectories))
            print("Collected branching points:", len(self.episode_buffer._branching_points))
            print("Preference pairs:", len(preference_dataset))
            print("Prompt message counts:", sorted(prompt_message_counts.items()))
            print("Example prompt tail:")
            example = preference_dataset[0]
            for message in example["prompt"][-4:]:
                print(message)
            print("Example chosen:", example["chosen"])
            print("Example rejected:", example["rejected"])

            if dump_pairs_path or save_pairs_path:
                import gzip as _gzip
                dump_path = Path(dump_pairs_path or save_pairs_path)
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                with _gzip.open(dump_path, "wt", encoding="utf-8") as f:
                    for row in preference_dataset:
                        f.write(json.dumps(dict(row)) + "\n")
                print(f"Dumped {len(preference_dataset)} preference pairs to {dump_path}")
                if dump_pairs_path:
                    return  # Dump-only mode; SAVE continues into DPO below.

        output_dir = Path(env_str("PLAYPEN_OUTPUT_DIR", f"models/dpo+lora/{self.learner.name}"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if load_pairs_path:
            summary = {
                "source": "off-policy",
                "load_pairs_path": load_pairs_path,
                "preference_pairs": len(preference_dataset),
            }
        else:
            summary = {
                "instances": len(dataset),
                "experiments": experiment_counts,
                "branching_factor": branching_factor,
                "min_round": min_round,
                "max_round": max_round,
                "temperature": branch_temperature,
                "max_tokens": branch_max_tokens,
                "valid_reply_retries": valid_reply_retries,
                "valid_reply_temp_step": valid_reply_temp_step,
                "valid_reply_min_turn": valid_reply_min_turn,
                "valid_reply_max_turn": valid_reply_max_turn,
                "reply_quality_weight": reply_quality_weight,
                "solver_top_k": solver_top_k,
                "solver_max_exact_pool": solver_max_exact_pool,
                "valid_reply_retry_stats": dict(sorted(retry_stats.items())),
                "records_dir": str(records_dir),
                "trajectories": len(self.episode_buffer._trajectories),
                "branching_points": len(self.episode_buffer._branching_points),
                "preference_pairs": len(preference_dataset),
                "prompt_message_counts": dict(sorted(prompt_message_counts.items())),
                "pair_reply_validity_raw": raw_pair_validity_summary,
                "pair_reply_validity_final": pair_validity_summary,
                "pair_selection_raw": raw_pair_selection_summary,
                "pair_selection_final": filtered_pair_selection_summary,
                "identical_reply_pairs_raw": raw_identical_pairs,
                "identical_reply_pairs_final": final_identical_pairs,
                "require_valid_chosen": require_valid_chosen,
                "require_valid_rejected": require_valid_rejected,
                "require_distinct_reply": require_distinct_reply,
                "min_score_gap": min_score_gap,
                "valid_words": len(valid_words),
            }
        (output_dir / "branching_preference_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        eval_ratio = env_float("PLAYPEN_EVAL_RATIO", 0.1)
        if len(preference_dataset) < 2:
            print("Preference dataset too small for train/eval split; disabling eval.")
            train_dataset = preference_dataset
            eval_dataset = None
            eval_strategy = "no"
        else:
            eval_size = max(1, math.ceil(len(preference_dataset) * eval_ratio))
            eval_size = min(eval_size, len(preference_dataset) - 1)
            split = preference_dataset.train_test_split(test_size=eval_size, shuffle=True, seed=seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            eval_strategy = "steps"
        print("Train pairs:", len(train_dataset))
        print("Eval pairs:", 0 if eval_dataset is None else len(eval_dataset))

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
            output_dir=str(output_dir),
            max_length=max_length,
            eval_strategy=eval_strategy,
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
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
        final_dir = output_dir / "final"
        trainer.save_model(str(final_dir))
        print("Saved final adapter to:", final_dir)
