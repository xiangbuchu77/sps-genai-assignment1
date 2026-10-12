"""Course FastAPI application: text generation, embeddings, and CIFAR-10 CNN."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, StringConstraints
import spacy

from app.bigram_model import BigramModel
from app.classifier import ImageClassifier, InvalidImage

MODEL_NAME = "en_core_web_lg"
Word = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Static word vectors do not require tagging, parsing, or entity recognition.
    app.state.nlp = spacy.load(
        MODEL_NAME,
        exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    app.state.classifier = ImageClassifier()
    yield
    del app.state.nlp
    del app.state.classifier


app = FastAPI(title="Applied Generative AI: Text and Image API", lifespan=lifespan)

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: Word
    length: int = Field(default=20, ge=1, le=200, strict=True)


class EmbeddingRequest(BaseModel):
    word: Word


class EmbeddingResponse(BaseModel):
    word: str
    model: str
    dimensions: int
    embedding: list[float]


class ImageClassificationResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    if len(request.start_word.split()) != 1:
        raise HTTPException(status_code=422, detail="start_word must be a single word.")
    return {"generated_text": bigram_model.generate_text(request.start_word, request.length)}


@app.post("/embedding", response_model=EmbeddingResponse)
def calculate_embedding(payload: EmbeddingRequest, request: Request):
    """Return the full static vector for one alphabetic spaCy token."""
    word = request.app.state.nlp(payload.word)
    if len(word) != 1 or not word[0].is_alpha:
        raise HTTPException(status_code=422, detail="Provide one alphabetic word, without punctuation.")
    if not word.has_vector:
        raise HTTPException(status_code=404, detail="This word has no vector in en_core_web_lg.")
    return EmbeddingResponse(
        word=payload.word,
        model=MODEL_NAME,
        dimensions=len(word.vector),
        embedding=word.vector.tolist(),
    )


@app.post("/classify", response_model=ImageClassificationResponse)
def classify_image(request: Request, file: UploadFile = File(...)):
    """Upload an image to predict one of the ten CIFAR-10 classes."""
    content = file.file.read(5 * 1024 * 1024 + 1)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be at most 5 MiB.")
    if not content:
        raise HTTPException(status_code=422, detail="Image file is empty.")
    try:
        return request.app.state.classifier.predict(content)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
