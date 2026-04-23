import * as ort from 'onnxruntime-web';
import { loadModel } from '../download-model.js';
import type { BBox, ModelLoadStatus } from '../types.js';
import { letterboxToTensor, MODEL_INPUT_SIZE } from './preprocess.js';
import { decodeDetections } from './postprocess.js';

let segmenterSession: ort.InferenceSession | null = null;

export const loadSegmenter = async (
  url: string, 
  providers: string[] = ['webgpu', 'wasm'],
  onProgress?: (status: ModelLoadStatus) => void
): Promise<ort.InferenceSession> => {
  if (!segmenterSession) {
    const modelBuffer = await loadModel(url, onProgress);
    segmenterSession = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: providers,
    });
  }

  return segmenterSession;
}

export const segmentImage = async (
  file: File,
  options: { segmenterUrl: string; executionProviders?: string[], maxDetections?: number } = { segmenterUrl: '' },
): Promise<Array<{ normalizedBounds: BBox; pxBounds: BBox, area: number }>> => {
  if (!options.segmenterUrl) throw new Error('segmenterUrl is required');

  const bitmap = await createImageBitmap(file);
  const segmenter = await loadSegmenter(options.segmenterUrl, options.executionProviders);

  const { tensor, scale, padX, padY } = letterboxToTensor(bitmap);
  const { width: origW, height: origH } = bitmap;

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const feeds = { [segmenter.inputNames[0]]: inputTensor };
  const output = await segmenter.run(feeds);

  const output0 = output[segmenter.outputNames[0]];
  const output1 = output[segmenter.outputNames[1]];

  const detections = decodeDetections(
    output0.data as Float32Array,
    output1.data as Float32Array,
    scale, padX, padY,
    origW, origH, 
    options.maxDetections
  );

  bitmap.close();
  return detections;
}