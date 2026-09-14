from __future__ import annotations

import math
from collections import Counter


def wordle_feedback_pattern(guess: str, target: str) -> str:
    if len(guess) != 5 or len(target) != 5:
        raise ValueError("guess and target must both be 5 letters")

    result = ["r"] * 5
    remaining = Counter(target)

    for idx, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            result[idx] = "g"
            remaining[g] -= 1

    for idx, g in enumerate(guess):
        if result[idx] == "g":
            continue
        if remaining[g] > 0:
            result[idx] = "y"
            remaining[g] -= 1

    return "".join(result)


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


def expected_reduction_score(guess: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    pattern_counts = Counter(wordle_feedback_pattern(guess, target) for target in candidates)
    total = float(len(candidates))
    expected_remaining = sum(count * count for count in pattern_counts.values()) / total
    return total - expected_remaining


def entropy_score(guess: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    pattern_counts = Counter(wordle_feedback_pattern(guess, target) for target in candidates)
    total = float(len(candidates))
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def top_k_guesses(candidates: list[str], k: int, max_exact_pool: int = 256) -> list[str]:
    if k <= 0 or not candidates:
        return []

    heuristic = candidate_quality_scores(candidates)
    if len(candidates) > max_exact_pool:
        candidate_pool = sorted(candidates, key=lambda word: (heuristic.get(word, 0.0), word), reverse=True)[:max_exact_pool]
    else:
        candidate_pool = list(candidates)

    scored = []
    for word in candidate_pool:
        reduction = expected_reduction_score(word, candidates)
        entropy = entropy_score(word, candidates)
        heuristic_score = heuristic.get(word, 0.0)
        scored.append((reduction, entropy, heuristic_score, word))

    scored.sort(reverse=True)
    return [word for _, _, _, word in scored[:k]]
