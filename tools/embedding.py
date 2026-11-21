"""Shared bge-m3 encoder used by build_index.py and query_index.py.

The model is BAAI/bge-m3 exported to ONNX (int8) by Xenova. Inference uses
onnxruntime, CLS pooling and L2 normalisation, matching the official
sentence-transformers config of bge-m3.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "bge-m3"

MODEL_NAME = "BAAI/bge-m3"
MODEL_URL = "https://huggingface.co/BAAI/bge-m3"
ONNX_URL = "https://huggingface.co/Xenova/bge-m3/resolve/main"
DIMENSION = 1024
MAX_TOKENS = 512

FILES = {
    "config.json": f"{ONNX_URL}/config.json",
    "tokenizer.json": f"{ONNX_URL}/tokenizer.json",
    "tokenizer_config.json": f"{ONNX_URL}/tokenizer_config.json",
    "special_tokens_map.json": f"{ONNX_URL}/special_tokens_map.json",
    "model_int8.onnx": f"{ONNX_URL}/onnx/model_int8.onnx",
}


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "quantumforge-rag-index/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def ensure_model(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Download the model files if they are missing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = model_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"downloading {name} ...")
        _download(url, target)
    return model_dir


class BgeM3:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        self.session = ort.InferenceSession(
            str(model_dir / "model_int8.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer.encode_batch(texts[start : start + batch_size])
            feed = {
                "input_ids": np.array([item.ids for item in batch], dtype=np.int64),
                "attention_mask": np.array([item.attention_mask for item in batch], dtype=np.int64),
            }
            hidden = self.session.run(None, feed)[0]
            cls = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            vectors.append(cls / np.clip(norms, 1e-12, None))
        return np.vstack(vectors)


def model_dir_from_env() -> Path:
    return Path(os.environ.get("BGE_M3_DIR", DEFAULT_MODEL_DIR))
