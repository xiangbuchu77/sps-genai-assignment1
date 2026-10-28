# Assignment 3: Image Generation

Jiajun Shen | js6890

GitHub: https://github.com/xiangbuchu77/sps-genai-assignment1/tree/assignment3-gan

- [Written answers and results](Assignment3_Submission.pdf)
- [Editable written answers](Assignment3_Answers.md)
- [GAN architecture](app/gan.py), [training](train_gan.py), and [image inference](app/digit_generator.py)
- [API](app/main.py), [training history](models/gan_training_history.json), and [run settings](models/gan_metrics.json)
- [Generated samples](examples/mnist_generated.png) and [Docker HTTP verification](models/gan_api_verification.json)

## Run the API

The trained checkpoint is included. No training or GPU is required for inference.

```bash
git clone --branch assignment3-gan --single-branch https://github.com/xiangbuchu77/sps-genai-assignment1.git
cd sps-genai-assignment1
docker build -t sps-genai-assignment3 .
docker run --rm -p 127.0.0.1:8000:80 sps-genai-assignment3
```

Open http://127.0.0.1:8000/docs. Use **POST /generate-image** with:

```json
{"num_samples":16,"seed":42}
```

The response is a grayscale PNG grid of generated 28 x 28 images:

```bash
curl -X POST http://127.0.0.1:8000/generate-image \
  -H 'Content-Type: application/json' \
  -d '{"num_samples":16,"seed":42}' --output generated_digits.png
```

`num_samples` is an integer from 1 to 64; `seed` is an integer from 0 to
4294967295. The headers `X-Num-Samples`, `X-Seed`, and `X-Model-Epoch` describe
the response. Requests use an independent random generator; the same seed
reproduces the same result on the same runtime. No digit class is specified:
this is an unconditional GAN.

The existing `GET /`, `POST /generate`, `POST /embedding`, and `POST /classify`
endpoints remain available. The image embeds the GAN and earlier CNN weights.
The first build needs Internet access for dependencies and the existing spaCy model.

To run locally with Python 3.12 and uv:

```bash
uv sync --frozen
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Required architecture

```text
Generator:
[B,100] -> Linear(100,6272) -> reshape [B,128,7,7]
-> ConvTranspose2d(128,64,4,2,1) -> BatchNorm2d(64) -> ReLU
-> ConvTranspose2d(64,1,4,2,1) -> Tanh -> [B,1,28,28]

Discriminator:
[B,1,28,28] -> Conv2d(1,64,4,2,1) -> LeakyReLU(0.2)
-> Conv2d(64,128,4,2,1) -> BatchNorm2d(128) -> LeakyReLU(0.2)
-> Flatten(6272) -> Linear(6272,1)
```

The generator has 765,761 parameters and the discriminator has 138,817.
D returns logits during training; sigmoid gives a real/fake probability.
`BCEWithLogitsLoss` includes sigmoid, so an additional sigmoid is not applied
before the training loss.

## Training

- Data: all 60,000 official MNIST training images; digit labels are ignored.
- Input normalization: pixel / 127.5 - 1, matching the generator's Tanh range.
- Latent noise: 100 independent standard-normal values per image.
- Optimizers: separate Adam optimizers, learning rate 0.0002, betas (0.5, 0.999).
- Batch size: 128; epochs: 30; seed: 42; device: Apple MPS.
- One D update and one G update per batch, including the final partial batch.
- D objective: BCE(real,1) + BCE(fake,0); G objective: BCE(D(G(z)),1).

Reproduce the run:

```bash
uv run python train_gan.py --epochs 30 --batch-size 128 --seed 42
```

Data downloads into `data/`. The script saves both networks in
`models/mnist_gan.pt`, per-epoch history, run settings, and a 64-image grid from
fixed noise seed 1234. Temporary per-epoch previews are written to `tmp/assignment3/`.
The checkpoint contains inference weights, not optimizer/RNG state for exact
training resumption. Retraining updates the checkpoint and examples; rebuild
Docker to include new weights. Results can vary across devices.

All 64 final samples are shown without filtering. Many resemble digits, while
some are ambiguous or distorted. No classifier accuracy, FID score, or full
coverage of the digit distribution is claimed. GAN losses are training signals,
not calibrated image-quality scores.

## Checks and written work

```bash
uv run pytest tests/test_gan.py tests/test_cnn.py -q
```

Checks cover the required layers and dimensions, gradients, adversarial updates,
trained checkpoint inference, deterministic sampling without changing global
random state or BatchNorm buffers, PNG/API agreement, and invalid requests.
They also exercise the earlier CNN and text/embedding routes. Numerical checks
verify the transposed-convolution and BatchNorm calculations in the written answers.

Docker was built and queried over HTTP without host-mounted checkpoints; the
saved PNG was compared with direct inference inside the same container.
Details are in `models/gan_api_verification.json`.

The seven theory answers and full calculations are in the linked PDF and
Markdown file. Earlier Assignment 1/2 documents and code remain available.
