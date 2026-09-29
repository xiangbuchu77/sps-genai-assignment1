"""Bigram sampling adapted from Module 2, Practice 2."""

from collections import Counter, defaultdict
import random
import re


class BigramModel:
    def __init__(self, corpus: list[str]):
        self.transitions = defaultdict(Counter)
        for text in corpus:
            words = re.findall(r"\b\w+\b", text.lower())
            for current, following in zip(words, words[1:]):
                self.transitions[current][following] += 1

    def generate_text(self, start_word: str, length: int) -> str:
        words = [start_word.lower()]
        for _ in range(length - 1):
            next_words = self.transitions.get(words[-1])
            if not next_words:
                break
            # Sampling weights normalize the observed continuation counts.
            words.append(random.choices(list(next_words), weights=list(next_words.values()))[0])
        return " ".join(words)
