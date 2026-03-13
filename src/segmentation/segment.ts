import * as ort from 'onnxruntime-web';
import { letterboxToTensor, MODEL_INPUT_SIZE } from './preprocess.js';
import { decodeDetections } from './postprocess.js';
import type { BBox } from '../types.js';

let segmenterSession: ort.InferenceSession | null = null;

const loadSegmenter = async (url: string, providers: string[] = ['wasm']): Promise<ort.InferenceSession> => {
  if (!segmenterSession)
    segmenterSession = await ort.InferenceSession.create(url, { executionProviders: providers });
  return segmenterSession;
}

export const segmentImage = async (
  file: File,
  options: { segmenterUrl: string; executionProviders?: string[] } = { segmenterUrl: '' }
): Promise<Array<{ bbox: BBox; area: number }>> => {
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
  );

  bitmap.close();
  return detections;
}