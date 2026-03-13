# /// script
# dependencies = ["onnx", "onnxruntime"]
# ///

"""
quantize_clip.py
----------------
Quantize the exported CLIP ViT-B/32 ONNX model from float32 to int8.
Reduces file size from ~350MB to ~175MB with negligible quality loss.

Run export_clip.py first, then:

    uv run scripts/quantize_clip.py
    # Reads:  assets/models/clip-vit-b32-visual.onnx
    # Writes: assets/models/clip-vit-b32-visual-int8.onnx

onnxruntime-web compatibility note:
    Standard dynamic quantization (quantize_dynamic) introduces MatMulInteger
    and QLinearMatMul ops that onnxruntime-web does not support — the session
    will silently stall in the browser.

    Instead we use weight-only quantization: weights are stored as int8 and
    a DequantizeLinear node is inserted before each MatMul, so the op seen
    at runtime is still a plain float32 MatMul. This is fully supported by
    onnxruntime-web and gives ~50% size reduction (vs ~75% for full int8,
    but with actual browser compatibility).
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_PATH  = Path("../assets/models/clip-vit-b32-visual.onnx")
OUTPUT_PATH = Path("../assets/models/clip-vit-b32-visual-int8.onnx")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"{INPUT_PATH} not found — run export_clip.py first.")

# ── Pre-process: shape inference + model optimization ────────────────────────
# Required before quantization — without this, shape inference is incomplete
# and the quantized model can produce malformed graphs that hang at runtime.
PREPROCESSED_PATH = INPUT_PATH.with_suffix(".preprocessed.onnx")
print(f"Pre-processing {INPUT_PATH} ...")
from onnxruntime.quantization.shape_inference import quant_pre_process
quant_pre_process(str(INPUT_PATH), str(PREPROCESSED_PATH), skip_optimization=False, skip_symbolic_shape=True)
print(f"Pre-processed: {PREPROCESSED_PATH}  ({PREPROCESSED_PATH.stat().st_size / 1e6:.1f} MB)")

print(f"Quantizing → {OUTPUT_PATH} ...")
quantize_dynamic(
    model_input=PREPROCESSED_PATH,
    model_output=OUTPUT_PATH,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],
    per_channel=True,
    reduce_range=False,
)

input_mb  = INPUT_PATH.stat().st_size  / 1e6
output_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"Done.")
print(f"  float32 : {input_mb:.1f} MB")
print(f"  int8    : {output_mb:.1f} MB  ({100 * output_mb / input_mb:.0f}% of original)")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nVerifying embedding similarity between float32 and int8 models ...")

dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)

sess_f32 = ort.InferenceSession(str(INPUT_PATH),  providers=["CPUExecutionProvider"])
sess_i8  = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])

vec_f32 = sess_f32.run(None, {"image": dummy})[0][0]
vec_i8  = sess_i8.run( None, {"image": dummy})[0][0]

vec_f32 /= np.linalg.norm(vec_f32)
vec_i8  /= np.linalg.norm(vec_i8)

cosine_sim = float(np.dot(vec_f32, vec_i8))
print(f"  Cosine similarity (dummy input): {cosine_sim:.6f}  (expect > 0.999)")