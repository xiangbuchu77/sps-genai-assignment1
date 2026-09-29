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

`uv` uses Python 3.12 and installs the pinned dependencies from `uv.lock`, including `en_core_web_lg` 3.8.0. The model download is approximately 382 MiB. Internet access is needed during installation; inference runs locally.

Open http://127.0.0.1:8000/docs for interactive API documentation.

## Docker deployment

Install and start Docker Desktop (or Docker Engine on Linux). From the repository root, run:

```bash
docker build -t sps-genai-assignment1 .
docker run --rm -d --name sps-genai-assignment1 -p 127.0.0.1:8000:80 sps-genai-assignment1
docker ps --filter name=sps-genai-assignment1
```

Wait until the container reports `healthy`, then open http://127.0.0.1:8000/docs. The host port is 8000 and the container port is 80. If port 8000 is occupied, change the mapping to `127.0.0.1:8001:80` and use port 8001 in the URLs.

The image installs the locked runtime dependencies and the full spaCy model during the build. No host Python installation, local virtual environment, model directory, or bind mount is needed. The application listens on `0.0.0.0` inside the container. The Docker build context includes only the application and dependency files. Allow several minutes and sufficient disk space for the first model download and build.

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H 'Content-Type: application/json' \
  -d '{"word":"apple"}'
docker logs sps-genai-assignment1
docker stop sps-genai-assignment1
```

The image was built and tested successfully on `linux/arm64`. The container reported `healthy`, and all eight live HTTP checks passed without host directory mounts. The `apple` embedding matched the locally verified vector exactly. `evidence/docker_verification.json` records the image ID and results. The Dockerfile follows the native architecture selected by Docker; other architectures have not been tested.

To repeat these checks while the container is running, use `python tools/verify_container.py` (Python 3, standard library only). This saves fresh verification evidence.

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
- `Dockerfile` and `.dockerignore`: container build, runtime command, and health check.

## Course and technical references

- Assignment1-1.pdf, Questions 1-6 and implementation requirements.
- Module_2_Practical_2_Word_Sampling-1.ipynb, bigram sampling.
- Module_2_Practical_3_Word_Embeddings-2.ipynb, `en_core_web_lg` and `calculate_embedding`.
- [Module 3 classroom FastAPI activity](https://gurgentus.github.io/applied_genai_notebooks/Module%203/gentext_project/).
- [spaCy: vectors and similarity](https://spacy.io/usage/linguistic-features#vectors-similarity).
- [FastAPI: testing](https://fastapi.tiangolo.com/tutorial/testing/).
- [uv: Docker integration](https://docs.astral.sh/uv/guides/integration/docker/).

Course files are referenced by title and are not redistributed in this repository.
