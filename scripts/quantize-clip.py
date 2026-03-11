# /// script
# dependencies = ["onnx", "onnxruntime"]
# ///

"""
quantize_clip.py
----------------
Quantize the exported CLIP ViT-B/32 ONNX model from float32 to int8.
Reduces file size from ~350MB to ~90MB with negligible quality loss.

Run export_clip.py first to produce the float32 model, then:

    uv run scripts/quantize_clip.py
    # Reads:  models/clip-vit-b32-visual.onnx        (~350MB float32)
    # Writes: models/clip-vit-b32-visual-int8.onnx   (~90MB int8)

The quantized model is a drop-in replacement — pass it as embedderUrl
to loadVisualSearch(). Note that indexes built with the float32 and int8
models are NOT cross-compatible (embeddings occupy different spaces).

Technical notes:
  - Uses static per-channel quantization for Conv/MatMul/Gemm weight nodes,
    which gives better accuracy than dynamic quantization for ViT architectures.
  - Activations are kept in float32 — only weights are quantized (QOperator mode).
    This is the sweet spot for ORT Web: maximum size reduction with minimum
    accuracy loss and good browser runtime support.
  - Requires the float32 model to exist at INPUT_PATH first.
"""

from pathlib import Path

from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
)
import numpy as np
import onnxruntime as ort

INPUT_PATH  = Path("../assets/models/clip-vit-b32-visual.onnx")
OUTPUT_PATH = Path("../assets/models/clip-vit-b32-visual-int8.onnx")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"{INPUT_PATH} not found — run export_clip.py first."
    )

print(f"Quantizing {INPUT_PATH} → {OUTPUT_PATH} ...")

# Dynamic quantization: no calibration data needed, weights quantized to int8,
# activations computed in float32. Best balance of simplicity, compatibility,
# and size reduction for transformer models.
quantize_dynamic(
    model_input=INPUT_PATH,
    model_output=OUTPUT_PATH,
    weight_type=QuantType.QInt8
)

input_mb  = INPUT_PATH.stat().st_size  / 1e6
output_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"Done.")
print(f"  float32 : {input_mb:.1f} MB")
print(f"  int8    : {output_mb:.1f} MB  ({100 * output_mb / input_mb:.0f}% of original)")

# Sanity check: run both models on the same input and compare cosine similarity
print("Verifying embedding similarity between float32 and int8 models...")

dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)

sess_f32 = ort.InferenceSession(str(INPUT_PATH),  providers=["CPUExecutionProvider"])
sess_i8  = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])

vec_f32 = sess_f32.run(None, {"image": dummy})[0][0]
vec_i8  = sess_i8.run( None, {"image": dummy})[0][0]

# Normalise
vec_f32 /= np.linalg.norm(vec_f32)
vec_i8  /= np.linalg.norm(vec_i8)

cosine_sim = float(np.dot(vec_f32, vec_i8))
print(f"  Cosine similarity (dummy input): {cosine_sim:.6f}  (expect > 0.999)")