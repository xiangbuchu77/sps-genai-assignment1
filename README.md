# Assignment 1

FastAPI word embeddings using spaCy's `en_core_web_lg` model. Based on the Module 2 notebooks and Module 3 class activity.

## Run with Docker

Start Docker, then run:

```bash
git clone https://github.com/xiangbuchu77/sps-genai-assignment1.git
cd sps-genai-assignment1
docker build -t sps-genai-assignment1 .
docker run --rm -d --name sps-genai-assignment1 \
  -p 127.0.0.1:8000:80 sps-genai-assignment1
```

The first build downloads the model and dependencies. Open http://127.0.0.1:8000/docs once the container is ready.

## Try the API

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H 'Content-Type: application/json' \
  -d '{"word":"apple"}'
```

Returns the word, model name, dimension, and all 300 vector values. Use one alphabetic token; invalid input returns 422, and words without a vector return 404.

The original routes are also available:
- `GET /`: returns `{"Hello": "World"}`.
- `POST /generate`: accepts `{"start_word":"the","length":20}`.

Stop the container with `docker stop sps-genai-assignment1`.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --frozen
uv run fastapi dev app/main.py
```

Run tests with `uv run pytest -q`. Docker was tested on linux/arm64.

## Probability questions

The written answers are in [Assignment1_Submission.pdf](output/pdf/Assignment1_Submission.pdf). To reproduce the calculations:

```bash
uv run python probability_solutions.py
```
