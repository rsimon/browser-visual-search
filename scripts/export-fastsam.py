# /// script
# dependencies = ["ultralytics", "onnx"]
# ///

"""
export_fastsam.py
-----------------
Export FastSAM-s to ONNX for use with onnxruntime-web.

Usage:
    uv run scripts/export_fastsam.py
    # Produces: models/fastsam-s.onnx

The exported model:
    Input:  images   [1, 3, 1024, 1024]  float32, normalised [0,1]
    Output: output0  [1, 37, 8400]       float32  (box + conf + mask coeffs)
            output1  [1, 32, 256, 256]   float32  (prototype masks)

Notes:
  - opset 12 is the highest reliably supported by onnxruntime-web as of 2025.
  - dynamic_axes on the batch dimension is omitted intentionally — ORT Web
    works better with fully static shapes.
  - Do NOT use simplify=True here; it can fold away the mask proto output.
"""

from pathlib import Path
from ultralytics import FastSAM

MODELS_DIR = Path("models")
ASSETS_DIR = Path("../assets/models")

INPUT_MODEL  = MODELS_DIR / "FastSAM-s.pt"
OUTPUT_MODEL = ASSETS_DIR / "fastsam-s.onnx"

print(f"Loading {INPUT_MODEL} ...")
model = FastSAM(str(INPUT_MODEL))

print("Exporting to ONNX ...")
model.export(
    format="onnx",
    imgsz=1024,
    opset=12,
    simplify=False,
    dynamic=False,
)

# Ultralytics saves alongside the .pt file — move to our preferred path
exported = INPUT_MODEL.with_suffix(".onnx")
if exported != OUTPUT_MODEL:
    exported.rename(OUTPUT_MODEL)

print(f"Exported: {OUTPUT_MODEL}  ({OUTPUT_MODEL.stat().st_size / 1e6:.1f} MB)")
print()
print("Output tensor names (verify these match postprocess.ts expectations):")

import onnx
m = onnx.load(str(OUTPUT_MODEL))
for out in m.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}  {shape}")