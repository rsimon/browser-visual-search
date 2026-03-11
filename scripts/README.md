# Preparing ONNX Models

## 1. FastSAM-s (segmenter)

```bash
# Download model weights (~23 MB)
curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt \
     -o models/FastSAM-s.pt

# Export to ONNX
uv run export-fastsam.py
# → ../assets/fastsam-s.onnx  (~45 MB)
```

## 2. CLIP ViT-B/32 (embedder)

CLIP weights are fetched automatically on first run — no manual download needed.

```bash
# Export to ONNX (float32)
uv run export-clip.py
# → ../assets/clip-vit-b32-visual.onnx  (~350 MB)

# Optional: quantize to int8 for a ~4× smaller file
uv run quantize-clip.py
# → ../assets/clip-vit-b32-visual-int8.onnx  (~89 MB)
```

### float32 vs int8

| Model | Size | Quality |
|---|---|---|
| `clip-vit-b32-visual.onnx` | ~350 MB | Full float32 precision |
| `clip-vit-b32-visual-int8.onnx` | ~90 MB | Negligible quality loss |

The int8 model is recommended for most use cases. Pass either as `embedderUrl`
to `loadVisualSearch()`.

> **Note:** Indexes built with the float32 and int8 models are not
> cross-compatible. Switching models requires re-indexing your collection.