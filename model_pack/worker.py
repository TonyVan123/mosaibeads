from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Load CUDA/cuDNN DLLs installed by the optional pip extras.  This keeps the
    # machine clean: no system-wide CUDA Toolkit is required.
    dll_handles = []
    if hasattr(os, "add_dll_directory"):
        site_packages = Path(ort.__file__).resolve().parent.parent
        dll_dirs = list((site_packages / "nvidia").glob("*/bin"))
        # cuDNN 9 delay-loads its split sublibraries with LoadLibrary, which on
        # Windows consults PATH even when the parent DLL came from add_dll_directory.
        os.environ["PATH"] = os.pathsep.join(map(str, dll_dirs)) + os.pathsep + os.environ.get("PATH", "")
        for dll_dir in dll_dirs:
            dll_handles.append(os.add_dll_directory(str(dll_dir)))
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except Exception:
            pass
    available = ort.get_available_providers()
    preferred = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    session = ort.InferenceSession(args.model, providers=preferred)
    batch = np.load(args.input).astype(np.float32)
    features = session.run(None, {session.get_inputs()[0].name: batch})[0]
    features = features.reshape(features.shape[0], -1)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    similarities = features[1:] @ features[0]
    scores = np.clip((similarities - 0.35) / 0.65, 0, 1).astype(float).tolist()
    payload = {"provider": session.get_providers()[0], "scores": scores}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
