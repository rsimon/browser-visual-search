# Export ONNX Models

## 1. FastSAM-s (segmenter)

```bash
# Download model weights
curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt \
     -o models/FastSAM-s.pt

# Place FastSAM-s.pt in ./models/ first (download from Ultralytics)
uv run export-fastsam.py
# → ../assets/fastsam-s.onnx  (~45 MB)
```

## 2. CLIP ViT-B/32 (embedder)

```bash
uv run export-clip.py
# → ../assets/clip-vit-b32-visual.onnx  (~350 MB)
# Note: model weights are fetched automatically
```

