/**
 * segment.ts
 * ----------
 * Provides a `Segmenter` factory that wraps a FastSAM ONNX model and
 * exposes a simple `segment(file)` / `segmentBitmap()` API.  The model is
 * loaded lazily on first use via a dynamic import of `onnxruntime-web`.
 *
 * Consumers may supply their own `ort.InferenceSession` (e.g. with a
 * custom model URL), making it easy to use a different segmentation model.
 */

import type { BBox } from './types.js';
import { letterboxToTensor } from './preprocess.js';
import { decodeDetections, RawDetection } from './postprocess.js';

export interface SegmenterOptions {
  /** URL to an ONNX model to load. */
  modelUrl?: string;

  /** Pre‑created ONNX runtime session; if provided, `modelUrl` is ignored. */
  session?: any; // will refine to ort.InferenceSession when imported

  /** Preferred execution providers (e.g. ['webgpu','wasm']). */
  executionProviders?: string[];
}

export interface SegmentResult {
  bbox: BBox;
  area: number;
}

export interface Segmenter {
  segment(file: File, options?: { signal?: AbortSignal }): Promise<SegmentResult[]>;
  segmentBitmap(bitmap: ImageBitmap): Promise<SegmentResult[]>;
}

export function createSegmenter(opts: SegmenterOptions = {}): Segmenter {
  let _session: any | null = null;

  async function getSession() {
    if (_session) return _session;
    const ort = await import('onnxruntime-web');
    if (opts.session) {
      _session = opts.session;
    } else {
      if (!opts.modelUrl) {
        throw new Error('Segmenter requires a modelUrl or a pre‑created session');
      }
      _session = await ort.InferenceSession.create(opts.modelUrl, {
        executionProviders: opts.executionProviders ?? ['wasm'],
      });
    }
    return _session;
  }

  async function segmentBitmap(bitmap: ImageBitmap): Promise<SegmentResult[]> {
    const session = await getSession();

    const { tensor, scale, padX, padY } = letterboxToTensor(bitmap);
    const { width: origW, height: origH } = bitmap;

    const ort = await import('onnxruntime-web');
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 1280, 1280]);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const output = await session.run(feeds);

    const output0 = output[session.outputNames[0]];
    const output1 = output[session.outputNames[1]];

    const detections = decodeDetections(
      output0.data as Float32Array,
      output1.data as Float32Array,
      scale, padX, padY,
      origW, origH,
    );

    return detections;
  }

  async function segment(file: File, options?: { signal?: AbortSignal }): Promise<SegmentResult[]> {
    options?.signal?.throwIfAborted?.();
    const bitmap = await createImageBitmap(file);
    try {
      const results = await segmentBitmap(bitmap);
      return results;
    } finally {
      bitmap.close();
    }
  }

  return { segment, segmentBitmap };
}
