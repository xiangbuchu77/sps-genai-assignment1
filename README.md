# Assignment 1

- [Written answers](Assignment1_Submission.pdf)
- [Probability code (Q1-Q6)](probability_solutions.py)
- [API code](app/main.py)

## Run

With Docker running, download this repository and run these commands from its folder:

```bash
docker build -t sps-genai-assignment1 .
docker run --rm -p 127.0.0.1:8000:80 sps-genai-assignment1
```

Open http://127.0.0.1:8000/docs. Try `POST /embedding` with `{"word":"apple"}` to get a 300-dimensional vector. The model downloads during the first build.

To run the probability calculations:

```bash
python3 probability_solutions.py
```

Based on the Module 2 notebooks and Module 3 FastAPI activity.
