# Export ONNX Models

## 1. FastSAM-s (segmenter)

```bash
# Download model weights
curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt \
     -o models/FastSAM-s.pt

# Place FastSAM-s.pt in ./models/ first (download from Ultralytics)
uv run export-fastsam.py
# → models/fastsam-s.onnx  (~40 MB)
```

## 2. CLIP ViT-B/32 (embedder)

```bash
uv run export-clip.py
# Note: model weights are fetched automatically
```

