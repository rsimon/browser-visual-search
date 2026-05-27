import { modelBoxToBBox } from './preprocess.js';
import type { BBox, ModelConfig } from '../types.js';

export interface RawDetection {
  normalizedBounds: BBox;
  pxBounds: BBox;
  area: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const iou = (
  ax1: number, ay1: number, ax2: number, ay2: number,
  bx1: number, by1: number, bx2: number, by2: number,
): number => {
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter === 0) return 0;
  return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter);
};

// ── NMS ───────────────────────────────────────────────────────────────────────

interface Proposal {
  x1: number; y1: number; x2: number; y2: number;
  conf: number;
  coeffs: Float32Array;
}

const nms = (proposals: Proposal[], iouThreshold: number): Proposal[] => {
  proposals.sort((a, b) => b.conf - a.conf);
  const suppressed = new Uint8Array(proposals.length);
  const kept: Proposal[] = [];

  for (let i = 0; i < proposals.length; i++) {
    if (suppressed[i]) continue;
    kept.push(proposals[i]);
    const p = proposals[i];
    for (let j = i + 1; j < proposals.length; j++) {
      if (!suppressed[j] &&
          iou(p.x1, p.y1, p.x2, p.y2,
              proposals[j].x1, proposals[j].y1, proposals[j].x2, proposals[j].y2) > iouThreshold) {
        suppressed[j] = 1;
      }
    }
  }

  return kept;
};

// ── Mask decoding ─────────────────────────────────────────────────────────────

const decodeMask = (
  coeffs: Float32Array,
  protos: Float32Array,
  protoSize: number,
  numProtos: number,
): Float32Array => {
  const n    = protoSize * protoSize;
  const mask = new Float32Array(n);

  for (let k = 0; k < numProtos; k++) {
    const c = coeffs[k], offset = k * n;
    for (let i = 0; i < n; i++) mask[i] += c * protos[offset + i];
  }
  for (let i = 0; i < n; i++) mask[i] = sigmoid(mask[i]);

  return mask;
};

const maskAreaInBBox = (
  mask: Float32Array,
  x1: number, y1: number, x2: number, y2: number,
  origW: number, origH: number,
  inputSize: number,
  protoSize: number,
  threshold = 0.5,
): number => {
  const s   = protoSize / inputSize;
  const mx1 = Math.max(0,         Math.floor(x1 * s));
  const my1 = Math.max(0,         Math.floor(y1 * s));
  const mx2 = Math.min(protoSize, Math.ceil (x2 * s));
  const my2 = Math.min(protoSize, Math.ceil (y2 * s));

  let positive = 0;
  for (let row = my1; row < my2; row++)
    for (let col = mx1; col < mx2; col++)
      if (mask[row * protoSize + col] > threshold) positive++;

  const protoPixelArea = (inputSize / protoSize) ** 2;
  return (positive * protoPixelArea) / (origW * origH);
};

// ── Public interface ───────────────────────────────────────────────────────────

const NUM_ATTRS  = 37;
const NUM_PROTOS = 32;

export const decodeDetections = (
  output0Data: Float32Array,
  output1Data: Float32Array,
  scale: number,
  padX: number,
  padY: number,
  origW: number,
  origH: number,
  config: ModelConfig,
  maxDetections = Infinity,
): RawDetection[] => {
  const { inputSize, protoSize, confThreshold, iouThreshold } = config;
  const numAnchors = output0Data.length / NUM_ATTRS;

  const proposals: Proposal[] = [];

  for (let a = 0; a < numAnchors; a++) {
    const conf = output0Data[4 * numAnchors + a];
    if (conf < confThreshold) continue;

    const cx = output0Data[0 * numAnchors + a];
    const cy = output0Data[1 * numAnchors + a];
    const w  = output0Data[2 * numAnchors + a];
    const h  = output0Data[3 * numAnchors + a];

    const coeffs = new Float32Array(NUM_PROTOS);
    for (let k = 0; k < NUM_PROTOS; k++)
      coeffs[k] = output0Data[(5 + k) * numAnchors + a];

    proposals.push({ x1: cx - w/2, y1: cy - h/2, x2: cx + w/2, y2: cy + h/2, conf, coeffs });
  }

  if (proposals.length === 0) return [];

  const kept = nms(proposals, iouThreshold).slice(0, maxDetections);

  return kept
    .map(p => {
      const mask = decodeMask(p.coeffs, output1Data, protoSize, NUM_PROTOS);
      const area = maskAreaInBBox(mask, p.x1, p.y1, p.x2, p.y2, origW, origH, inputSize, protoSize);
      const { normalizedBounds, pxBounds } = modelBoxToBBox(
        p.x1, p.y1, p.x2, p.y2, scale, padX, padY, origW, origH,
      );
      return { normalizedBounds, pxBounds, area };
    })
    .sort((a, b) => b.area - a.area);
};