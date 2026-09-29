# Assignment 1: FastAPI Word Embeddings and Probability

This project extends the Module 3 text-generation API with the spaCy word-embedding functionality from Module 2. The probability solutions are included as a PDF and a reproducible Python script.

## Setup and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/xiangbuchu77/sps-genai-assignment1.git
cd sps-genai-assignment1
uv sync --frozen
uv run fastapi dev app/main.py
```

`uv` uses Python 3.12 and installs the pinned dependencies from `uv.lock`, including `en_core_web_lg` 3.8.0. The model download is approximately 382 MiB. Internet access is needed during installation; inference runs locally. Docker is optional in the assignment and is not required here.

Open http://127.0.0.1:8000/docs for interactive API documentation.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Original classroom response: `{"Hello": "World"}` |
| POST | `/generate` | Sample text from the classroom bigram corpus |
| POST | `/embedding` | Return the complete 300-dimensional word vector |

### Query a word embedding

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H 'Content-Type: application/json' \
  -d '{"word":"apple"}'
```

The response includes `word`, `model`, `dimensions`, and `embedding` (all 300 floating-point values). An actual response is saved in `evidence/embedding_apple.json`.

The endpoint implements the classroom operation `nlp(input_word).vector`, converts the NumPy vector to a JSON list, and loads the model once at startup. Unneeded linguistic pipeline components are excluded; the tokenizer and pretrained static word vectors remain available. Case is preserved and surrounding whitespace is trimmed.

Input must be one alphabetic spaCy token of 1-100 characters. Empty input, phrases, punctuation, numbers, and incorrectly typed fields return HTTP 422. A valid word missing from the model vocabulary returns HTTP 404 instead of a misleading zero vector. Hyphenated words and contractions that tokenize into multiple tokens are rejected by this single-word interface.

### Generate text

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"start_word":"the","length":20}'
```

`length` is the maximum total word count, including the starting word (1-200). Generation stops early if a word has no observed continuation. Sampling is random, so outputs can differ. The small corpus is copied from the Module 3 example; bigrams are counted within each corpus entry, without artificial transitions between entries. Relative continuation counts are used as sampling weights.

## Validate and reproduce

```bash
uv run pytest -q
uv run python probability_solutions.py
```

Tests use the actual installed spaCy model and cover exact agreement with its vectors, vector dimensions, invalid requests, out-of-vocabulary words, and the existing text-generation behavior. `evidence/verification.json` records the live HTTP checks and package versions. `evidence/probability_results.json` contains the calculation results.

## Files

- `app/main.py`: API routes, input validation, model lifecycle.
- `app/bigram_model.py`: bigram sampling adapted from Module 2.
- `probability_solutions.py`: calculations for all six questions.
- `solutions.md`: editable written solutions.
- `output/pdf/Assignment1_Submission.pdf`: combined implementation report and worked solutions.
- `tests/test_api.py`: integration tests.
- `uv.lock`: resolved dependency versions for reproducible installation.

## Course and technical references

- Assignment1-1.pdf, Questions 1-6 and implementation requirements.
- Module_2_Practical_2_Word_Sampling-1.ipynb, bigram sampling.
- Module_2_Practical_3_Word_Embeddings-2.ipynb, `en_core_web_lg` and `calculate_embedding`.
- [Module 3 classroom FastAPI activity](https://gurgentus.github.io/applied_genai_notebooks/Module%203/gentext_project/).
- [spaCy: vectors and similarity](https://spacy.io/usage/linguistic-features#vectors-similarity).
- [FastAPI: testing](https://fastapi.tiangolo.com/tutorial/testing/).

Course files are referenced by title and are not redistributed in this repository.
