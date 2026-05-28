import * as ort from 'onnxruntime-web';
import { loadModel } from '../download-model.js';
import type { BBox, ModelLoadStatus } from '../types.js';
import { letterboxToTensor } from './preprocess.js';
import { decodeDetections } from './postprocess.js';
import { getModelName, MODEL_CONFIGURATIONS, ModelName } from '../model-configurations.js';

let segmenterSession: { modelUrl: string, session: ort.InferenceSession } | null = null;

export const loadSegmenter = async (
  url: string,
  providers: string[] = ['webgpu', 'wasm'],
  onProgress?: (status: ModelLoadStatus) => void,
): Promise<ort.InferenceSession> => {
  if (segmenterSession?.modelUrl !== url) {
    segmenterSession?.session?.release();

    const modelBuffer = await loadModel(url, onProgress);
    const session  = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: providers,
    });

    segmenterSession = { session, modelUrl: url };
  }

  return segmenterSession?.session;
}

const resolveConfig = (segmenterUrl: string) => {
  const modelName = getModelName(segmenterUrl);
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
  const { segmenterUrl, executionProviders, maxDetections } = options;
  const config   = resolveConfig(segmenterUrl);

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