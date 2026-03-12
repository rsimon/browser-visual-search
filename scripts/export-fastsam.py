# /// script
# dependencies = ["ultralytics", "onnx"]
# ///

"""
export_fastsam.py
-----------------
Export FastSAM-s to ONNX for use with onnxruntime-web.

Usage:
    uv run scripts/export_fastsam.py
    # Produces: assets/models/fastsam-s.onnx
"""

from pathlib import Path
from ultralytics import FastSAM

MODELS_DIR = Path("models")
ASSETS_DIR = Path("../assets/models")
MODELS_DIR.mkdir(exist_ok=True)

INPUT_MODEL  = MODELS_DIR / "FastSAM-s.pt"
OUTPUT_MODEL = ASSETS_DIR / "fastsam-s.onnx"

print(f"Loading {INPUT_MODEL} ...")
model = FastSAM(str(INPUT_MODEL))

print("Exporting to ONNX ...")
model.export(
    format="onnx",
    imgsz=1280,
    opset=12,
    simplify=False,
    dynamic=False,
    max_det=800,
)

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