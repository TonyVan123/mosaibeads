from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


pack = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="beadsketch_verify_") as folder:
    input_path = Path(folder) / "input.npy"
    output_path = Path(folder) / "output.json"
    np.save(input_path, np.zeros((2, 3, 224, 224), dtype=np.float32))
    command = [sys.executable, str(pack / "worker.py"), "--model",
               str(pack / "semantic_encoder.onnx"), "--input", str(input_path),
               "--output", str(output_path)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if completed.returncode:
        print(completed.stderr[-1000:])
        raise SystemExit(completed.returncode)
    provider = json.loads(output_path.read_text(encoding="utf-8"))["provider"]
    print(f"Actual inference provider: {provider}")
    raise SystemExit(0 if provider == "CUDAExecutionProvider" else 2)
