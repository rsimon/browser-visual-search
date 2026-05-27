import * as ort from 'onnxruntime-web';
import { loadModel } from '../download-model.js';
import type { BBox, ModelLoadStatus } from '../types.js';
import { letterboxToTensor } from './preprocess.js';
import { decodeDetections } from './postprocess.js';
import { MODEL_CONFIGURATIONS, ModelName } from '../model-configurations.js';

let segmenterSession: ort.InferenceSession | null = null;

export const loadSegmenter = async (
  url: string,
  providers: string[] = ['webgpu', 'wasm'],
  onProgress?: (status: ModelLoadStatus) => void,
): Promise<ort.InferenceSession> => {
  if (!segmenterSession) {
    const startTime = performance.now();
    console.log('[browser-visual-search] Loading segmenter');

    const modelBuffer = await loadModel(url, onProgress);
    console.log(`[browser-visual-search] Done (${Math.round(performance.now() - startTime)} ms)`);
    
    console.log('[browser-visual-search] Creating session');
    segmenterSession  = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: providers,
    });
    console.log('[browser-visual-search] Session done');
  }
  return segmenterSession;
};

const resolveConfig = (segmenterUrl: string) => {
  const modelName = segmenterUrl.split('/').pop()?.replace(/\.onnx$/, '') ?? '';
  if (!(modelName in MODEL_CONFIGURATIONS)) {
    throw new Error(
      `No config found for model "${modelName}". ` +
      `Add an entry to MODEL_CONFIGS in models.ts.`
    );
  }
  return MODEL_CONFIGURATIONS[modelName as ModelName];
};

export const segmentImage = async (
  file: File,
  options: {
    segmenterUrl: string;
    executionProviders?: string[];
    maxDetections?: number;
  },
): Promise<Array<{ normalizedBounds: BBox; pxBounds: BBox; area: number }>> => {
  console.log('[browser-visual-search] segmentImage');
  console.log(options);

  const { segmenterUrl, executionProviders, maxDetections } = options;

  const config   = resolveConfig(segmenterUrl);
  console.log('[browser-visual-search] resolved config');
  console.log(config);

  const bitmap   = await createImageBitmap(file);
  const segmenter = await loadSegmenter(segmenterUrl, executionProviders);

  const { tensor, scale, padX, padY } = letterboxToTensor(bitmap, config.inputSize);
  const { width: origW, height: origH } = bitmap;

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, config.inputSize, config.inputSize]);
  const feeds       = { [segmenter.inputNames[0]]: inputTensor };
  const output      = await segmenter.run(feeds);

  const detections = decodeDetections(
    output[segmenter.outputNames[0]].data as Float32Array,
    output[segmenter.outputNames[1]].data as Float32Array,
    scale, padX, padY, origW, origH,
    config,
    maxDetections,
  );

  bitmap.close();
  return detections;
}