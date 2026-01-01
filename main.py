import base64
import io
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from transformers import AutoModel
from huggingface_hub import hf_hub_download

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN
from PIL import Image
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from torchvision import transforms

load_dotenv()

def _get_env(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        return default if default is not None else ""
    return value


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


QDRANT_URL = _get_env("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = _get_env("QDRANT_API_KEY", "my-secret-key")
QDRANT_COLLECTION = _get_env("QDRANT_COLLECTION", "faces")
FACE_MATCH_THRESHOLD = _get_float("FACE_MATCH_THRESHOLD", 0.6)
FACE_SEARCH_TOPK = _get_int("FACE_SEARCH_TOPK", 5)
FACE_VECTOR_SIZE = _get_int("FACE_VECTOR_SIZE", 512)

ADAFACE_MODEL_ID = _get_env("ADAFACE_MODEL_ID", "minchul/cvlface_adaface_ir101_ms1mv3")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FaceRegistrationRequest(BaseModel):
    employee_id: str = Field(..., min_length=1, description="Unique employee identifier")
    employee_name: str = Field(..., min_length=1, description="Full name of the employee")
    image_base64: str = Field(..., description="Base64 encoded RGB face image (without data URL prefix)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FaceRecognitionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded RGB face image (without data URL prefix)")
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class FaceImageProcessor:
    def __init__(self, device: str) -> None:
        self.detector = MTCNN(
            keep_all=False,
            device=device,
            margin=10,
            image_size=160,
            post_process=True,
        )

    def extract_face(self, image: Image.Image) -> Optional[Image.Image]:
        face_tensor = self.detector(image)
        if face_tensor is None:
            return None
        if isinstance(face_tensor, torch.Tensor):
            face_tensor = (
                face_tensor.permute(1, 2, 0)
                .mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )
        return Image.fromarray(face_tensor)


class AdaFaceEmbedder:
    """
    Embedder using AdaFace IR101 MS1MV3 model from HuggingFace.
    Downloads the repository locally, adds it to sys.path, then loads via
    AutoModel.from_pretrained(local_path, trust_remote_code=True).
    """

    def __init__(self, device: str, model_id: str) -> None:
        self.device = device
        self.repo_id = model_id
        print(f"[AdaFaceEmbedder] Loading model from HuggingFace: {model_id}")

        cache_dir = os.path.expanduser(
            os.path.join("~", ".cvlface_cache", "minchul", "cvlface_adaface_ir101_ms1mv3")
        )

        self._download_repo(self.repo_id, cache_dir)
        self.model = self._load_model_from_local_path(cache_dir)

        self.model.to(device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _download_repo(self, repo_id: str, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        files_path = os.path.join(path, "files.txt")
        if not os.path.exists(files_path):
            hf_hub_download(
                repo_id,
                "files.txt",
                local_dir=path,
                local_dir_use_symlinks=False,
            )
        with open(files_path, "r") as f:
            files = f.read().split("\n")

        for file in [f for f in files if f] + ["config.json", "wrapper.py", "model.safetensors"]:
            full_path = os.path.join(path, file)
            if not os.path.exists(full_path):
                hf_hub_download(
                    repo_id,
                    file,
                    local_dir=path,
                    local_dir_use_symlinks=False,
                )

    def _load_model_from_local_path(self, path: str):
        cwd = os.getcwd()
        os.chdir(path)
        sys.path.insert(0, path)
        try:
            model = AutoModel.from_pretrained(path, trust_remote_code=True)
        finally:
            sys.path.pop(0)
            os.chdir(cwd)
        return model

    def create_embedding(self, face_image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(face_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        if isinstance(outputs, dict):
            if "embeddings" in outputs:
                embedding = outputs["embeddings"]
            else:
                embedding = next(iter(outputs.values()))
        else:
            embedding = outputs

        if embedding.ndim == 4:
            embedding = torch.flatten(embedding, 1)

        embedding = torch.nn.functional.normalize(embedding, dim=1)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)


class QdrantVectorStore:
    def __init__(self, url: str, api_key: Optional[str], collection: str, vector_size: int) -> None:
        self.collection = collection
        self.vector_size = vector_size
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(col.name == self.collection for col in collections):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def upsert(self, point_id: str, vector: np.ndarray, payload: Dict[str, Any]) -> None:
        point = qmodels.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=payload,
        )
        self.client.upsert(collection_name=self.collection, points=[point])

    def delete(self, point_id: str) -> None:
        selector = qmodels.PointIdsList(points=[point_id])
        self.client.delete(
            collection_name=self.collection,
            points_selector=selector,
        )

    def search(self, vector: np.ndarray, limit: int) -> List[Any]:
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                limit=limit,
                with_payload=True,
            )
            points = getattr(response, "points", response)
            return list(points)
        raise RuntimeError("Qdrant client does not support querying points")


class FaceService:
    def __init__(
        self,
        processor: FaceImageProcessor,
        embedder: AdaFaceEmbedder,
        store: QdrantVectorStore,
        default_threshold: float,
        default_top_k: int,
    ) -> None:
        self.processor = processor
        self.embedder = embedder
        self.store = store
        self.default_threshold = default_threshold
        self.default_top_k = default_top_k

    def register(self, request: FaceRegistrationRequest) -> Dict[str, Any]:
        face_image = self._image_to_face(request.image_base64)
        embedding = self.embedder.create_embedding(face_image)
        point_id = str(uuid.uuid4())
        payload = {
            "employee_id": request.employee_id,
            "employee_name": request.employee_name,
            "metadata": request.metadata,
        }
        self.store.upsert(point_id=point_id, vector=embedding, payload=payload)
        return {"point_id": point_id, "employee_id": request.employee_id}

    def recognize(self, request: FaceRecognitionRequest) -> Dict[str, Any]:
        threshold = request.threshold or self.default_threshold
        top_k = request.top_k or self.default_top_k
        face_image = self._image_to_face(request.image_base64)
        embedding = self.embedder.create_embedding(face_image)
        results = self.store.search(vector=embedding, limit=top_k)
        if not results:
            return {"matched": False, "reason": "no_vectors"}
        best = results[0]
        score = float(best.score)
        matched = score >= threshold
        response = {
            "matched": matched,
            "confidence": score,
            "threshold": threshold,
            "candidate": {
                "point_id": best.id,
                "payload": best.payload,
            },
        }
        if not matched:
            return response
        response["employee_id"] = best.payload.get("employee_id")
        response["employee_name"] = best.payload.get("employee_name")
        return response

    def delete(self, point_id: str) -> Dict[str, Any]:
        self.store.delete(point_id=point_id)
        return {"deleted": True, "point_id": point_id}

    def _image_to_face(self, image_base64: str) -> Image.Image:
        image = decode_base64_image(image_base64)
        face = self.processor.extract_face(image)
        if face is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        debug_dir = Path("tmp")
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4()}.jpg"
        face.save(debug_dir / filename)
        
        return face


def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Provided data is not a valid image") from exc


image_processor = FaceImageProcessor(device=DEVICE)
face_embedder = AdaFaceEmbedder(device=DEVICE, model_id=ADAFACE_MODEL_ID)
vector_store = QdrantVectorStore(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection=QDRANT_COLLECTION,
    vector_size=FACE_VECTOR_SIZE,
)
face_service = FaceService(
    processor=image_processor,
    embedder=face_embedder,
    store=vector_store,
    default_threshold=FACE_MATCH_THRESHOLD,
    default_top_k=FACE_SEARCH_TOPK,
)

app = FastAPI(title="Face enrollment service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    vector_store.ensure_collection()


@app.post("/face/register")
def register_face(request: FaceRegistrationRequest) -> Dict[str, Any]:
    result = face_service.register(request)
    return {
        "status": "registered",
        "employee_id": result["employee_id"],
        "point_id": result["point_id"],
    }


@app.post("/face/recognize")
def recognize_face(request: FaceRecognitionRequest) -> Dict[str, Any]:
    return face_service.recognize(request)


@app.delete("/face/{point_id}")
def delete_face(point_id: str) -> Dict[str, Any]:
    return face_service.delete(point_id)

@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE,
        "collection": QDRANT_COLLECTION,
        "threshold": FACE_MATCH_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
