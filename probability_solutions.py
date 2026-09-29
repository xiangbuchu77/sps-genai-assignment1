"""Reproduce all six probability exercises using the Python standard library."""

import json
from math import log2


def solve():
    scores = [85, 90, 95, 100]
    probabilities = [0.375, 0.375, 0.125, 0.125]
    sample = [85, 90, 85, 95, 90, 85, 100, 90]
    expected = sum(x * p for x, p in zip(scores, probabilities))
    second_moment = sum(x ** 2 * p for x, p in zip(scores, probabilities))
    message_probs = [0.4, 0.3, 0.2, 0.1]
    return {
        "Q1": {"intersection": 0.4 * 0.3, "union": 0.4 + 0.3 - 0.4 * 0.3},
        "Q2": {"independent": 0.7 == 0.5, "intersection": 0.7 * 0.4,
               "product_of_marginals": 0.5 * 0.4},
        "Q3": {"P_B": 0.5 * 0.6 + 0.2 * 0.4,
               "P_A_given_B": 0.5 * 0.6 / (0.5 * 0.6 + 0.2 * 0.4)},
        "Q4": {"P_positive": 0.95 * 0.02 + 0.10 * 0.98,
               "P_disease_given_positive": 0.95 * 0.02 / (0.95 * 0.02 + 0.10 * 0.98)},
        "Q5": {"expected_value": expected, "second_moment": second_moment,
               "variance": second_moment - expected ** 2, "sample_mean": sum(sample) / len(sample)},
        "Q6": {"entropy_bits": -sum(p * log2(p) for p in message_probs),
               "uniform_entropy_bits": -sum(0.25 * log2(0.25) for _ in range(4))},
    }


if __name__ == "__main__":
    print(json.dumps(solve(), indent=2))
