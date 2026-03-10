# /// script
# dependencies = ["open-clip-torch", "torch", "onnx", "onnxruntime"]
# ///

"""
export_clip.py
--------------
Export the CLIP ViT-B/32 visual encoder to ONNX for use with onnxruntime-web.

IMPORTANT: The model name and pretrained weights here must stay in sync with
embedder.py (MODEL_NAME = "ViT-B-32", PRETRAINED = "openai"). Do not change
these without also updating the server-side embedder and re-indexing.

Usage:
    uv run scripts/export_clip.py
    # Produces: models/clip-vit-b32-visual.onnx

The exported model:
    Input:  image    [1, 3, 224, 224]  float32, normalised with CLIP mean/std
    Output: output   [1, 512]          float32, un-normalised embedding
                                       (L2 normalisation applied in TypeScript)

Notes:
  - Only the visual encoder is exported (not the text encoder).
  - We use the legacy TorchScript-based exporter (dynamo=False) rather than
    the new dynamo/onnxscript exporter. It honours opset_version directly,
    always produces a single self-contained file, and has well-established
    onnxruntime-web compatibility.
  - opset 14 is sufficient for ViT-B/32 with the legacy exporter.
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="QuickGELU mismatch")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import onnxruntime as ort
import open_clip
import torch

MODEL_NAME  = "ViT-B-32"
PRETRAINED  = "openai"
INPUT_SIZE  = 224
OPSET       = 14

MODELS_DIR  = Path("models")
OUTPUT_PATH = MODELS_DIR / "clip-vit-b32-visual.onnx"

MODELS_DIR.mkdir(exist_ok=True)

print(f"Loading {MODEL_NAME} ({PRETRAINED}) ...")
model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model.eval()

visual = model.visual
dummy  = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)

print(f"Exporting to {OUTPUT_PATH} (opset {OPSET}, legacy exporter) ...")
torch.onnx.export(
    visual,
    dummy,
    str(OUTPUT_PATH),
    dynamo=False,          # use legacy TorchScript-based exporter
    opset_version=OPSET,
    input_names=["image"],
    output_names=["output"],
    dynamic_axes=None,     # static batch=1; ORT Web prefers static shapes
)

size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"Exported: {OUTPUT_PATH}  ({size_mb:.1f} MB)")

print("Verifying ...")
sess = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
out  = sess.run(None, {"image": dummy.numpy()})
vec  = out[0]
assert vec.shape == (1, 512), f"Unexpected output shape: {vec.shape}"
norm = float(np.linalg.norm(vec[0]))
print(f"Output shape:   {vec.shape}  ✓")
print(f"Embedding norm: {norm:.4f}  (un-normalised; TypeScript will L2-normalise)")
print("Done.")