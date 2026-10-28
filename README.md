# Assignment 2: Convolutional Neural Networks

Jiajun Shen | js6890

GitHub: https://github.com/xiangbuchu77/sps-genai-assignment1/tree/assignment2-cnn

- [Written answers and results](Assignment2_Submission.pdf)
- [CNN architecture](app/cnn.py) and [training](train_cnn.py)
- [API](app/main.py) and [image inference](app/classifier.py)
- [Training history](models/training_history.json) and [test metrics](models/metrics.json)

## Run with Docker

```bash
git clone --branch assignment2-cnn --single-branch https://github.com/xiangbuchu77/sps-genai-assignment1.git
cd sps-genai-assignment1
docker build -t sps-genai-assignment2 .
docker run --rm -p 127.0.0.1:8000:80 sps-genai-assignment2
```

Open http://127.0.0.1:8000/docs. Use `POST /classify` to upload an image.
The image is converted to RGB, resized to 64 x 64, and normalized with mean/std
0.5. The response contains `class_id`, `class_name`, `confidence` (a softmax
score), and all ten class probabilities. The checkpoint is included, so no
training or GPU is required to run the API. The first Docker build needs Internet
access for Python dependencies and the existing spaCy model.

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F "file=@examples/cifar10_test_0000.png"
```

The example is test image 0 (true class: cat). Its actual Docker response is
saved in [examples/prediction_response.json](examples/prediction_response.json).
The classifier accepts files up to 5 MiB and 16 million pixels. It predicts only
the ten CIFAR-10 classes; its scores are not calibrated for arbitrary photos.

The previous `/`, `/generate`, and `/embedding` endpoints still work.

## Model and training

The required architecture is:

```text
RGB 64x64 -> Conv(3,16,3,s=1,p=1) -> ReLU -> MaxPool(2,2)
         -> Conv(16,32,3,s=1,p=1) -> ReLU -> MaxPool(2,2)
         -> Flatten(8192) -> Linear(8192,100) -> ReLU -> Linear(100,10)
```

The model has 825,398 parameters and returns logits during training.

- Data: 45,000 training images, 5,000 validation images, 10,000 official test images.
- Split and training seed: 42. Only training images receive random cropping and horizontal flipping.
- Optimizer: AdamW, initial learning rate 0.001, weight decay 0.0001.
- Batch size: 256; epochs: 15; cosine learning-rate schedule; Apple MPS.
- Best validation checkpoint: epoch 15, validation accuracy **64.52%**.
- Final held-out test accuracy: **65.18%**; cross-entropy loss: **0.9886**.

The checkpoint was selected using validation accuracy, then evaluated once on
the test set. Metrics and the checkpoint checksum are saved in `models/metrics.json`.

To retrain (Python 3.12 and uv):

```bash
uv sync --frozen
uv run python train_cnn.py --epochs 15 --batch-size 256 --seed 42
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The dataset downloads into `data/`. Training writes the best checkpoint,
per-epoch metrics, test results, and split indices into `models/`. The script
chooses CUDA, MPS, or CPU automatically; use `--device cpu` to override it.
Exact results can vary across devices. If you retrain, rebuild the Docker image
to use the new weights; the submitted PDF describes the supplied checkpoint.

## Checks

```bash
uv run pytest tests/test_cnn.py -q
```

Checks cover the specified layer shapes and gradients, real checkpoint/API
agreement, grayscale/RGBA conversion, and invalid uploads. The Docker container
was also started and queried for the classifier and all three existing endpoints.

[Assignment 1 written answers](Assignment1_Submission.pdf) and
[probability code](probability_solutions.py) remain in the repository.
