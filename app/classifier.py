"""Load the trained checkpoint and classify an uploaded image on CPU."""

from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from app.cnn import AssignmentCNN, CLASSES, image_transform

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "cifar10_cnn.pt"


class InvalidImage(ValueError):
    pass


class ImageClassifier:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["classes"] != list(CLASSES) or checkpoint["image_size"] != 64:
            raise ValueError("Checkpoint class order or image size is incompatible.")
        self.model = AssignmentCNN()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = image_transform()

    def predict(self, content: bytes):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as source:
                    if source.width * source.height > 16_000_000:
                        raise InvalidImage("Image must contain no more than 16 million pixels.")
                    source.load()
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                    tensor = self.transform(rgb).unsqueeze(0)
        except (UnidentifiedImageError, OSError, ValueError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InvalidImage("Upload a valid image with no more than 16 million pixels.") from exc
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        index = int(probabilities.argmax())
        return {"class_id": index, "class_name": CLASSES[index],
                "confidence": float(probabilities[index]),
                "probabilities": {name: float(p) for name, p in zip(CLASSES, probabilities)}}
