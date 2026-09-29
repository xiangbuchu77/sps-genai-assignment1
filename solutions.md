# Assignment 1

## Part 1: FastAPI implementation

Repository: https://github.com/xiangbuchu77/sps-genai-assignment1

The application preserves the Module 3 `GET /` and `POST /generate` routes and adds `POST /embedding`. The new endpoint accepts a JSON object such as `{"word": "apple"}` and returns the queried word, model name, vector dimension, and all 300 vector values.

The implementation uses the same `en_core_web_lg` model and `nlp(input_word).vector` operation as the Module 2 word-embedding notebook. The model is loaded once at startup. Its NumPy vector is converted to a list so that FastAPI can serialize the response as JSON. Input is restricted to one alphabetic token. Invalid input returns HTTP 422; a word without a stored vector returns HTTP 404.

To install and run, execute `uv sync --frozen` followed by `uv run fastapi dev app/main.py`. Interactive documentation is available at http://127.0.0.1:8000/docs. Run `uv run pytest -q` for integration checks and `uv run python probability_solutions.py` to reproduce the calculations below. Docker deployment is optional and is not used.

## Part 2: Rules of Probability

### Question 1

Given P(A) = 0.4 and P(B) = 0.3, with A and B independent:

(a) Independence gives P(A ∩ B) = P(A)P(B) = 0.4 × 0.3 = **0.12**.

(b) The addition rule gives P(A ∪ B) = P(A) + P(B) − P(A ∩ B) = 0.4 + 0.3 − 0.12 = **0.58**.

### Question 2

Given P(A) = 0.5, P(B) = 0.4, and P(A | B) = 0.7:

Independence would require P(A | B) = P(A), since P(B) > 0. However, 0.7 ≠ 0.5. Therefore, **A and B are not independent**.

Equivalently, P(A ∩ B) = P(A | B)P(B) = 0.7 × 0.4 = 0.28, while P(A)P(B) = 0.5 × 0.4 = 0.20. These are unequal.

### Question 3

Given P(A) = 0.6, P(B | A) = 0.5, and P(B | Aᶜ) = 0.2:

P(Aᶜ) = 1 − 0.6 = 0.4.

By the law of total probability:

P(B) = P(B | A)P(A) + P(B | Aᶜ)P(Aᶜ) = 0.5 × 0.6 + 0.2 × 0.4 = 0.30 + 0.08 = 0.38.

By Bayes' rule:

P(A | B) = P(B | A)P(A) / P(B) = 0.30 / 0.38 = 15/19 ≈ **0.789474**, or **78.95%**.

### Question 4

Let D denote having the disease and + denote a positive test result. The given rates are P(D) = 0.02, P(+ | D) = 0.95, and P(− | Dᶜ) = 0.90.

Thus, P(Dᶜ) = 0.98 and the false-positive rate is P(+ | Dᶜ) = 1 − 0.90 = 0.10.

P(+) = P(+ | D)P(D) + P(+ | Dᶜ)P(Dᶜ) = 0.95 × 0.02 + 0.10 × 0.98 = 0.019 + 0.098 = 0.117.

P(D | +) = P(+ | D)P(D) / P(+) = 0.019 / 0.117 = 19/117 ≈ **0.162393**, or **16.24%**.

For example, among 10,000 people, the expected counts are 200 with the disease and 9,800 without it. There would be 190 true positives and 980 false positives. Consequently, 190 / (190 + 980) ≈ 16.24% of positive results would be true positives. The low disease prevalence explains why this is much lower than the 95% true-positive rate.

### Question 5

| x | P(X = x) | xP(X = x) | x²P(X = x) |
| --- | --- | --- | --- |
| 85 | 0.375 | 31.875 | 2709.375 |
| 90 | 0.375 | 33.750 | 3037.500 |
| 95 | 0.125 | 11.875 | 1128.125 |
| 100 | 0.125 | 12.500 | 1250.000 |
| Total | 1.000 | 90.000 | 8125.000 |

(a) E[X] = Σ xP(X = x) = 31.875 + 33.750 + 11.875 + 12.500 = **90**.

(b) E[X²] = Σ x²P(X = x) = 8125.

Var(X) = E[X²] − (E[X])² = 8125 − 90² = 8125 − 8100 = **25** (score points squared).

(c) For the sample {85, 90, 85, 95, 90, 85, 100, 90}:

Sample mean = (85 + 90 + 85 + 95 + 90 + 85 + 100 + 90) / 8 = 720 / 8 = **90**.

The sample mean equals E[X] for this particular sample. Its relative frequencies are 3/8, 3/8, 1/8, and 1/8, exactly matching the specified probabilities. Other random samples need not have exactly the same mean.

### Question 6

(a) With probabilities (0.4, 0.3, 0.2, 0.1), entropy in bits is:

H(X) = −Σ p(x)log₂ p(x)

= −[0.4 log₂(0.4) + 0.3 log₂(0.3) + 0.2 log₂(0.2) + 0.1 log₂(0.1)]

= 0.528771 + 0.521090 + 0.464386 + 0.332193

≈ **1.846439 bits**.

(b) If the four messages are equally likely, each has probability 1/4:

H(X) = −4 × (1/4)log₂(1/4) = −log₂(1/4) = **2 bits**.

The uniform distribution has the maximum entropy for four possible messages. Making the messages equally likely increases the entropy by approximately 2 − 1.846439 = **0.153561 bits**.
