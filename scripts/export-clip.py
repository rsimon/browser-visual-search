# /// script
# dependencies = ["open-clip-torch", "torch", "onnx", "onnxruntime"]
# ///

"""
export_clip.py
--------------
Export the CLIP ViT-B/32 visual encoder to ONNX for use with onnxruntime-web.

Usage:
    uv run scripts/export_clip.py
    # Produces: assets/models/clip-vit-b32-visual.onnx

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

ASSETS_DIR  = Path("../assets/models")
OUTPUT_PATH = ASSETS_DIR / "clip-vit-b32-visual.onnx"

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
    dynamo=False,
    opset_version=OPSET,
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={
        "image":  {0: "batch"},
        "output": {0: "batch"},
    },
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