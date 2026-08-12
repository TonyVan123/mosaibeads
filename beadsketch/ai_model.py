from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .palettes import resource_root


@dataclass(frozen=True)
class AIModelStatus:
    available: bool
    provider: str
    detail: str
    model_path: Path | None = None


def model_pack_dir() -> Path:
    # MOSAIBeads is the public V3 name. Keep the V2 variable as a compatibility
    # alias for existing GPU model-pack installations.
    override = os.environ.get("MOSAIBEADS_MODEL_PACK") or os.environ.get("BEADSKETCH_MODEL_PACK")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "model_pack"
    return resource_root() / "model_pack"


def _model_path() -> Path:
    return model_pack_dir() / "semantic_encoder.onnx"


def _worker_python() -> Path:
    return model_pack_dir() / ".venv" / "Scripts" / "python.exe"


def inspect_ai_model() -> AIModelStatus:
    model = _model_path()
    if not model.exists():
        return AIModelStatus(False, "未安装", "可选 AI 模型包不存在，智能调参仍可正常使用")
    worker = _worker_python()
    if worker.exists() and (model_pack_dir() / "worker.py").exists():
        return AIModelStatus(True, "ONNX Runtime（自动 GPU/CPU）",
                             "优先使用 NVIDIA CUDA，失败时自动回退 CPU", model)
    return AIModelStatus(True, "OpenCV DNN CPU", "模型已安装；GPU 运行时尚未安装", model)


def _prepare(rgb: np.ndarray) -> np.ndarray:
    image = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    image = (image - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32)
    return np.transpose(image, (2, 0, 1))


def _cosine_scores(features: np.ndarray) -> list[float]:
    features = features.reshape(features.shape[0], -1).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    similarities = features[1:] @ features[0]
    # MobileNet feature cosine usually occupies a narrow high range. Stretch it into
    # a useful ranking signal while retaining [0, 1] semantics.
    return np.clip((similarities - 0.35) / 0.65, 0, 1).astype(float).tolist()


class OpenCVDNNSemanticBackend:
    name = "AI 模型：MobileNetV3 / OpenCV DNN CPU"

    def __init__(self, model: Path):
        self.model = model
        # OpenCV's Windows filename path is not Unicode-safe on every build. Loading
        # bytes first also supports Chinese project folders reliably.
        self.net = cv2.dnn.readNetFromONNX(np.fromfile(model, dtype=np.uint8))

    def score(self, source_rgb: np.ndarray, candidates_rgb: list[np.ndarray]) -> list[float]:
        batch = np.stack([_prepare(source_rgb), *(_prepare(x) for x in candidates_rgb)])
        self.net.setInput(batch)
        return _cosine_scores(self.net.forward())


class WorkerSemanticBackend:
    def __init__(self, model: Path, python: Path):
        self.model = model
        self.python = python
        self.name = "AI 模型：ONNX Runtime（自动选择设备）"

    def score(self, source_rgb: np.ndarray, candidates_rgb: list[np.ndarray]) -> list[float]:
        batch = np.stack([_prepare(source_rgb), *(_prepare(x) for x in candidates_rgb)])
        with tempfile.TemporaryDirectory(prefix="beadsketch_ai_") as folder:
            input_path = Path(folder) / "input.npy"
            output_path = Path(folder) / "output.json"
            np.save(input_path, batch)
            command = [str(self.python), str(model_pack_dir() / "worker.py"),
                       "--model", str(self.model), "--input", str(input_path),
                       "--output", str(output_path)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=90,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-800:] or "AI worker failed")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.name = f"AI 模型：MobileNetV3 / {payload['provider']}"
            return [float(x) for x in payload["scores"]]


def load_semantic_backend(prefer_gpu: bool = True):
    status = inspect_ai_model()
    if not status.available or status.model_path is None:
        return None
    if prefer_gpu and _worker_python().exists() and (model_pack_dir() / "worker.py").exists():
        return WorkerSemanticBackend(status.model_path, _worker_python())
    return OpenCVDNNSemanticBackend(status.model_path)
