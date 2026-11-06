# Browser Visual Search

In-browser visual search powered by FastSAM, CLIP, onnxruntime-web and the File System API.

## Installation
```bash
npm install browser-visual-search
```

## Usage
```ts
import { buildIndex, loadIndex } from 'browser-visual-search';

// Build (and auto-save) an index from a directory
const dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });

const index = await buildIndex(dirHandle, {
  segmenterUrl: 'https://huggingface.co/aboutgeo/immarkus/resolve/main/fastsam-s.onnx',
  embedderUrl:  'https://huggingface.co/aboutgeo/immarkus/resolve/main/clip-vit-b32-visual.onnx',
  onProgress: (p) => console.log(p),
});

// Load a previously built index
const index = await loadIndex(dirHandle, {
  embedderUrl: 'https://huggingface.co/aboutgeo/immarkus/resolve/main/clip-vit-b32-visual.onnx',
});

// Query by image or region
const results = await index.query(file, /* bbox? */ [0.1, 0.1, 0.5, 0.5], { topK: 20 });
// → [{ imageId, bbox, area, score }, ...]
```

The index is persisted automatically to `.visual-search/` inside the chosen directory. Subsequent `buildIndex` calls are incremental — only new images are processed.

## Models

ONNX models tested with this package are available on [HuggingFace](https://huggingface.co/aboutgeo/immarkus/tree/main). 

You can also build models yourself: see [scripts/README](scripts) for instructions.

## Development

```bash
npm install
```

**Unit tests** (Vitest, runs in browser via Playwright):
```bash
npm test
```

Tests expect model files at `/assets/models/`.

**Test pages** — start a dev server (e.g. Vite) and open:

| Page | Purpose |
|---|---|
| `/test.html` | Build an index, inspect segments, run per-image embedding |
| `/search.html` | Load a saved index and run visual queries |