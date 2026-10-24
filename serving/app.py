import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import torch
import torch.nn as nn

# Model Definition
class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        return self.fc(out), hidden


app = FastAPI(
    title="OpenShift AI Character Completion Model Service",
    description="KServe v2 & REST compliant model serving runtime for PyTorch CharLSTM",
    version="1.0.0"
)

# Global model state
MODEL = None
CHAR2IDX = {}
IDX2CHAR = {}
VOCAB_SIZE = 0
MODEL_SOURCE = "uninitialized"


def load_model():
    global MODEL, CHAR2IDX, IDX2CHAR, VOCAB_SIZE, MODEL_SOURCE

    # Look in common model locations (S3 download dir, KServe /mnt/models, local artifacts)
    search_dirs = [
        "/tmp/models",
        os.getenv("MODEL_DIR", "/mnt/models"),
        "artifacts",
        "../artifacts",
        "/app/artifacts"
    ]

    # Try downloading latest trained model directly from in-cluster S3/MinIO
    s3_endpoint = os.getenv("S3_ENDPOINT", "http://minio-dspa:9000")
    if s3_endpoint:
        try:
            import boto3
            from botocore.client import Config
            s3_access = os.getenv("AWS_ACCESS_KEY_ID", "minio")
            s3_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "miniopassword")
            s3_client = boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access,
                aws_secret_access_key=s3_secret,
                config=Config(signature_version="s3v4")
            )
            Path("/tmp/models").mkdir(parents=True, exist_ok=True)
            s3_client.download_file("mlpipeline", "models/char-completion/model.pt", "/tmp/models/model.pt")
            s3_client.download_file("mlpipeline", "models/char-completion/vocab.json", "/tmp/models/vocab.json")
            print("[S3] Successfully downloaded latest model checkpoint from MinIO.")
        except Exception as s3_err:
            print(f"[INFO] S3 direct download skipped: {s3_err}")

    loaded = False
    for s_dir in search_dirs:
        weights_path = Path(s_dir) / "model.pt"
        vocab_path = Path(s_dir) / "vocab.json"

        # Also search recursively in case KServe mounted subdirectories
        if not weights_path.exists() and Path(s_dir).exists():
            matches = list(Path(s_dir).rglob("model.pt"))
            if matches:
                weights_path = matches[0]
                vocab_path = weights_path.parent / "vocab.json"

        if weights_path.exists() and vocab_path.exists():
            print(f"Loading model weights from: {weights_path}")
            with open(vocab_path) as f:
                vocab_data = json.load(f)
            CHAR2IDX = vocab_data["char2idx"]
            IDX2CHAR = {int(k): v for k, v in vocab_data["idx2char"].items()}
            VOCAB_SIZE = len(CHAR2IDX)

            MODEL = CharLSTM(vocab_size=VOCAB_SIZE)
            MODEL.load_state_dict(torch.load(weights_path, map_location="cpu"))
            MODEL.eval()
            MODEL_SOURCE = str(weights_path)
            loaded = True
            print(f"[OK] Model loaded successfully from {weights_path}")
            break

    if not loaded:
        print("[WARNING] No model artifact found in search paths. Initializing fallback vocabulary model...")
        words = ["openshift", "kubernetes", "kubeflow", "pipeline", "pytorch", "container", "kserve"]
        unique_chars = sorted(list(set("".join(words) + "$ ")))
        CHAR2IDX = {c: i for i, c in enumerate(unique_chars)}
        IDX2CHAR = {i: c for i, c in enumerate(unique_chars)}
        VOCAB_SIZE = len(CHAR2IDX)
        MODEL = CharLSTM(vocab_size=VOCAB_SIZE)
        MODEL.eval()
        MODEL_SOURCE = "in-memory-fallback"


@app.on_event("startup")
def startup_event():
    load_model()


def complete(prefix: str, max_length: int = 25) -> str:
    if not MODEL:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    prefix_clean = prefix.lower().strip()
    output_chars = list(prefix_clean)
    hidden = None

    with torch.no_grad():
        for char in prefix_clean:
            if char not in CHAR2IDX:
                return prefix_clean
            token = torch.tensor([[CHAR2IDX[char]]], dtype=torch.long)
            logits, hidden = MODEL(token, hidden)

        last_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        next_char = IDX2CHAR.get(last_token, "$")

        while next_char != "$" and len(output_chars) < max_length:
            output_chars.append(next_char)
            token = torch.tensor([[last_token]], dtype=torch.long)
            logits, hidden = MODEL(token, hidden)
            last_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            next_char = IDX2CHAR.get(last_token, "$")

    return "".join(output_chars)


# --- Root & Health Endpoints ---

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
@app.get("/v2/health/live")
@app.get("/v2/health/ready")
def health_check():
    """Liveness and readiness probes for KServe and Kubernetes."""
    return {
        "status": "ready",
        "model_loaded": MODEL is not None,
        "model_source": MODEL_SOURCE,
        "vocab_size": VOCAB_SIZE
    }


@app.get("/v2/models/char-completion")
def model_metadata():
    """KServe v2 Data Plane Model Metadata Endpoint."""
    return {
        "name": "char-completion",
        "versions": ["v1"],
        "platform": "PyTorch",
        "inputs": [
            {"name": "prefix", "datatype": "BYTES", "shape": [-1]}
        ],
        "outputs": [
            {"name": "completion", "datatype": "BYTES", "shape": [-1]}
        ]
    }


# --- Protocol 1: Friendly REST Endpoint ---

class CompletionRequest(BaseModel):
    prefix: str = Field(..., example="ope", description="Input character prefix to complete")
    max_length: Optional[int] = Field(25, example=25, description="Maximum length of completed word")


class CompletionResponse(BaseModel):
    prefix: str
    completion: str
    model: str = "char-completion"
    source: str


@app.post("/complete", response_model=CompletionResponse, tags=["Friendly REST API"])
def api_complete(request: CompletionRequest):
    """
    User-friendly REST endpoint for quick testing via curl or browser.
    Example: POST {"prefix": "ope"} -> {"completion": "openshift"}
    """
    start = time.time()
    result = complete(request.prefix, request.max_length or 25)
    return CompletionResponse(
        prefix=request.prefix,
        completion=result,
        model="char-completion",
        source=MODEL_SOURCE
    )


# --- Protocol 2: KServe v2 Data Plane Standard ---

class KServeInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: List[Any]


class KServeInferRequest(BaseModel):
    id: Optional[str] = None
    inputs: List[KServeInput]


class KServeOutput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: List[Any]


class KServeInferResponse(BaseModel):
    model_name: str
    model_version: Optional[str] = "v1"
    id: Optional[str] = None
    outputs: List[KServeOutput]


@app.post("/v2/models/char-completion/infer", response_model=KServeInferResponse, tags=["KServe v2 Data Plane"])
def kserve_infer(request: KServeInferRequest):
    """
    Enterprise-standard KServe v2 Data Plane protocol.
    Accepts standard KServe multi-input arrays and returns standard KServe tensor responses.
    """
    prefix_data = []
    for inp in request.inputs:
        if inp.name == "prefix" and inp.data:
            prefix_data = inp.data
            break

    if not prefix_data and request.inputs:
        prefix_data = request.inputs[0].data

    if not prefix_data:
        raise HTTPException(status_code=400, detail="Missing 'prefix' in KServe inputs")

    completions = [complete(str(p)) for p in prefix_data]

    return KServeInferResponse(
        model_name="char-completion",
        model_version="v1",
        id=request.id,
        outputs=[
            KServeOutput(
                name="completion",
                shape=[len(completions)],
                datatype="BYTES",
                data=completions
            )
        ]
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
