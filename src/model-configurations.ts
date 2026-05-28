import { ModelConfig } from './types.js';

export const MODEL_CONFIGURATIONS = {
  'fastsam-s': {
    inputSize:     1280,
    protoSize:     256,
    confThreshold: 0.01,
    iouThreshold:  0.5,
  },
  'yolo8_shapes_only': {
    inputSize:     800,
    protoSize:     200,
    confThreshold: 0.25,
    iouThreshold:  0.45,
  },
} as const satisfies Record<string, ModelConfig>;

export type ModelName = keyof typeof MODEL_CONFIGURATIONS;

export const getModelName = (url: string) => 
    url.split('/').pop()?.replace(/\.onnx$/, '') ?? '';