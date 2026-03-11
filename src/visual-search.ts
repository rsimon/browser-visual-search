/**
 * browser-visual-search
 * ─────────────────────
 * Main public API. Call loadVisualSearch() once at startup;
 * all other methods live on the returned object.
 *
 * Example:
 *   const vs = await loadVisualSearch({
 *     segmenterUrl: '/models/fastsam-s.onnx',
 *     embedderUrl:  '/models/clip-vit-b32-visual.onnx',
 *   });
 *
 *   // Index a collection
 *   const index = await vs.buildIndex(imageInputs, { onProgress });
 *   await index.serialize(dirHandle);
 *
 *   // Query
 *   const results = await vs.search(queryFile, queryBBox, index);
 */

import * as ort from 'onnxruntime-web';

import { fileToImageBitmap, letterboxToTensor, MODEL_INPUT_SIZE } from './preprocess.js';
import { decodeDetections } from './postprocess.js';
import { createIndexBuilder, deserializeIndex, MODEL_LABEL } from './index-store.js';
import { nearestNeighbours } from './search.js';

import type { BBox, BuildIndexOptions, IndexImageOptions, ImageInput, LoadOptions, SearchOptions, SearchResult, Segment } from './types.js';
import type { VisualSearchIndex } from './index-store.js';

export type { BBox, BuildIndexOptions, IndexImageOptions, ImageInput, LoadOptions, SearchOptions, SearchResult, Segment, VisualSearchIndex };
export { deserializeIndex };

const EMBEDDING_DIM = 512;
const CLIP_INPUT_SIZE = 224;

// ── CLIP preprocessing ────────────────────────────────────────────────────────

/**
 * Crop a region from a bitmap and produce a normalised [1,3,224,224] tensor
 * for the CLIP visual encoder.
 *
 * Uses OpenAI CLIP normalisation: mean=[0.48145466, 0.4578275, 0.40821073],
 * std=[0.26862954, 0.26130258, 0.27577711]
 */
function cropToClipTensor(
  bitmap: ImageBitmap,
  bbox: BBox | null, // null = full image
): Float32Array {
  const { width: W, height: H } = bitmap;

  let sx = 0, sy = 0, sw = W, sh = H;
  if (bbox) {
    sx = Math.round(bbox[0] * W);
    sy = Math.round(bbox[1] * H);
    sw = Math.max(1, Math.round(bbox[2] * W));
    sh = Math.max(1, Math.round(bbox[3] * H));
  }

  const S = CLIP_INPUT_SIZE;
  const canvas = new OffscreenCanvas(S, S);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, S, S);
  const { data } = ctx.getImageData(0, 0, S, S);

  const MEAN = [0.48145466, 0.4578275, 0.40821073];
  const STD  = [0.26862954, 0.26130258, 0.27577711];

  const tensor = new Float32Array(3 * S * S);
  const plane = S * S;
  for (let i = 0; i < plane; i++) {
    tensor[i]           = (data[i * 4]     / 255 - MEAN[0]) / STD[0]; // R
    tensor[plane + i]   = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1]; // G
    tensor[plane*2 + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2]; // B
  }
  return tensor;
}

/**
 * L2-normalise a Float32Array in place. Returns the same array.
 */
function l2Normalise(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

// ── The main VS object ────────────────────────────────────────────────────────

export interface VisualSearch {
  /**
   * Segment an image and embed each segment.
   * Analogous to POST /index-image on the server.
   */
  indexImage(input: ImageInput, progress?: IndexImageOptions): Promise<Segment[]>;

  /**
   * Embed a single image region (or the whole image if bbox is omitted).
   * Used for query-side embedding.
   */
  embedRegion(file: File, bbox?: BBox): Promise<Float32Array>;

  /**
   * Build a full in-memory index from a collection of images.
   */
  buildIndex(inputs: ImageInput[], options?: BuildIndexOptions): Promise<VisualSearchIndex>;

  /**
   * Embed a query region and return the top-K nearest segments from the index.
   */
  search(
    file: File,
    bbox: BBox | undefined,
    index: VisualSearchIndex,
    options?: SearchOptions,
  ): Promise<SearchResult[]>;
}

// ── loadVisualSearch ──────────────────────────────────────────────────────────

/**
 * Load both ONNX models and return the VisualSearch API object.
 * Call once at application startup.
 */
export async function loadVisualSearch(options: LoadOptions): Promise<VisualSearch> {
  const providers = options.executionProviders ?? ['wasm'];

  const [segmenter, embedder] = await Promise.all([
    ort.InferenceSession.create(options.segmenterUrl, { executionProviders: providers }),
    ort.InferenceSession.create(options.embedderUrl,  { executionProviders: providers }),
  ]);

  // ── Internal: run segmenter on a bitmap ────────────────────────────────────

  async function segmentBitmap(bitmap: ImageBitmap): Promise<Array<{ bbox: BBox; area: number }>> {
    const { tensor, scale, padX, padY } = letterboxToTensor(bitmap);
    const { width: origW, height: origH } = bitmap;

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
    const feeds = { [segmenter.inputNames[0]]: inputTensor };
    const output = await segmenter.run(feeds);

    const output0 = output[segmenter.outputNames[0]];
    const output1 = output[segmenter.outputNames[1]];

    console.log('output0 shape:', output0.dims);
    console.log('output names:', segmenter.outputNames);
    // Sample first anchor's values
    const d = output0.data as Float32Array;
    const N = d.length / 37;
    console.log('anchors:', N);
    console.log('first anchor cx,cy,w,h,conf:', d[0], d[N], d[2*N], d[3*N], d[4*N]);

    return decodeDetections(
      output0.data as Float32Array,
      output1.data as Float32Array,
      scale, padX, padY,
      origW, origH,
    );
  }

  // ── Internal: embed a single crop ────────────────────────────────────────────

  async function embedBitmap(bitmap: ImageBitmap, bbox: BBox | null): Promise<Float32Array> {
    const tensor = cropToClipTensor(bitmap, bbox);
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE]);
    const feeds = { [embedder.inputNames[0]]: inputTensor };
    const output = await embedder.run(feeds);
    const raw = output[embedder.outputNames[0]].data as Float32Array;
    return l2Normalise(new Float32Array(raw));
  }

  // ── Internal: embed multiple crops in batches ─────────────────────────────

  async function embedBitmapBatch(
    bitmap: ImageBitmap,
    bboxes: Array<BBox | null>,
    batchSize = 32,
  ): Promise<Float32Array[]> {
    const results: Float32Array[] = [];

    for (let i = 0; i < bboxes.length; i += batchSize) {
      const batchBBoxes = bboxes.slice(i, i + batchSize);
      const tensors = batchBBoxes.map(bbox => cropToClipTensor(bitmap, bbox));

      // Stack into a single [B, 3, 224, 224] tensor
      const B = tensors.length;
      const flat = new Float32Array(B * 3 * CLIP_INPUT_SIZE * CLIP_INPUT_SIZE);
      tensors.forEach((t, j) => flat.set(t, j * t.length));

      const inputTensor = new ort.Tensor('float32', flat, [B, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE]);
      const feeds = { [embedder.inputNames[0]]: inputTensor };
      const output = await embedder.run(feeds);
      const raw = output[embedder.outputNames[0]].data as Float32Array;

      // Split [B, 512] output back into individual vectors
      for (let j = 0; j < B; j++) {
        const vec = new Float32Array(EMBEDDING_DIM);
        for (let k = 0; k < EMBEDDING_DIM; k++) {
          vec[k] = raw[j * EMBEDDING_DIM + k];
        }
        results.push(l2Normalise(vec));
      }
    }

    return results;
  }

  // ── Public methods ─────────────────────────────────────────────────────────

  async function indexImage(input: ImageInput, progress?: IndexImageOptions): Promise<Segment[]> {
    const bitmap     = await fileToImageBitmap(input.file);
    const detections = await segmentBitmap(bitmap);

    progress?.onSegmentationDone?.(detections.length);

    const embeddings = await embedBitmapBatch(bitmap, detections.map(d => d.bbox));

    progress?.onEmbeddingDone?.(detections.length);

    const segments: Segment[] = detections.map((det, i) => ({
      bbox:      det.bbox,
      area:      det.area,
      embedding: embeddings[i],
    }));

    bitmap.close();
    return segments;
  }

  async function embedRegion(file: File, bbox?: BBox): Promise<Float32Array> {
    const bitmap = await fileToImageBitmap(file);
    const embedding = await embedBitmap(bitmap, bbox ?? null);
    bitmap.close();
    return embedding;
  }

  async function buildIndex(
    inputs: ImageInput[],
    options: BuildIndexOptions = {},
  ): Promise<VisualSearchIndex> {
    const builder = createIndexBuilder();
    const { onProgress, imageProgress } = options;

    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const segments = await indexImage(input, imageProgress);
      builder.addImage(input.id, segments);
      onProgress?.(i + 1, inputs.length);
    }

    return builder.build();
  }

  async function search(
    file: File,
    bbox: BBox | undefined,
    index: VisualSearchIndex,
    options: SearchOptions = {},
  ): Promise<SearchResult[]> {
    const queryEmbedding = await embedRegion(file, bbox);
    return nearestNeighbours(queryEmbedding, index, options.topK ?? 20);
  }

  return { indexImage, embedRegion, buildIndex, search };
}