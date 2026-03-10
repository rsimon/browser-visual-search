/**
 * Post-processing for FastSAM-s / YOLOv8-seg ONNX output.
 *
 * The model produces two output tensors:
 *
 *   output0  [1, 37, 8400]
 *            For each of 8400 anchors:
 *              [0..3]   cx, cy, w, h  (model-input pixel space)
 *              [4]      objectness confidence
 *              [5..36]  32 mask coefficients
 *
 *   output1  [1, 32, 256, 256]
 *            32 prototype masks (model-input space, scaled down 4×)
 *
 * Pipeline:
 *   1. Filter proposals by confidence threshold
 *   2. Convert cx/cy/w/h → x1/y1/x2/y2
 *   3. Non-maximum suppression (NMS)
 *   4. For each surviving detection: decode mask = sigmoid(coeffs @ protos)
 *   5. Compute mask area, convert bbox to normalised image coords
 */

import { modelBoxToNormalisedBBox } from './preprocess.js';
import type { BBox } from './types.js';

export interface RawDetection {
  /** Absolute pixel bbox in original image space, normalised [x,y,w,h] */
  bbox: BBox;
  /** Normalised mask area in [0,1] relative to original image */
  area: number;
}

const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD  = 0.5;
const PROTO_SIZE     = 256; // output1 spatial dimension
const NUM_PROTOS     = 32;

// ── Helpers ──────────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function intersectionArea(
  ax1: number, ay1: number, ax2: number, ay2: number,
  bx1: number, by1: number, bx2: number, by2: number,
): number {
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  return Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
}

function iou(
  ax1: number, ay1: number, ax2: number, ay2: number,
  bx1: number, by1: number, bx2: number, by2: number,
): number {
  const inter = intersectionArea(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2);
  if (inter === 0) return 0;
  const aArea = (ax2 - ax1) * (ay2 - ay1);
  const bArea = (bx2 - bx1) * (by2 - by1);
  return inter / (aArea + bArea - inter);
}

// ── NMS ──────────────────────────────────────────────────────────────────────

interface Proposal {
  x1: number; y1: number; x2: number; y2: number;
  conf: number;
  coeffs: Float32Array; // length 32
}

function nms(proposals: Proposal[], iouThreshold: number): Proposal[] {
  // Sort descending by confidence
  proposals.sort((a, b) => b.conf - a.conf);

  const kept: Proposal[] = [];
  const suppressed = new Uint8Array(proposals.length);

  for (let i = 0; i < proposals.length; i++) {
    if (suppressed[i]) continue;
    kept.push(proposals[i]);
    const p = proposals[i];
    for (let j = i + 1; j < proposals.length; j++) {
      if (suppressed[j]) continue;
      const q = proposals[j];
      if (iou(p.x1, p.y1, p.x2, p.y2, q.x1, q.y1, q.x2, q.y2) > iouThreshold) {
        suppressed[j] = 1;
      }
    }
  }

  return kept;
}

// ── Mask decoding ─────────────────────────────────────────────────────────────

/**
 * Decode a single instance mask from 32 coefficients and the prototype tensor.
 *
 * protos: flat Float32Array of shape [32, 256, 256], row-major.
 * Returns a flat Float32Array of shape [256, 256] with values in [0,1].
 */
function decodeMask(coeffs: Float32Array, protos: Float32Array): Float32Array {
  const protoSize = PROTO_SIZE * PROTO_SIZE;
  const mask = new Float32Array(protoSize);

  for (let k = 0; k < NUM_PROTOS; k++) {
    const c = coeffs[k];
    const offset = k * protoSize;
    for (let i = 0; i < protoSize; i++) {
      mask[i] += c * protos[offset + i];
    }
  }

  // Apply sigmoid
  for (let i = 0; i < protoSize; i++) {
    mask[i] = sigmoid(mask[i]);
  }

  return mask;
}

/**
 * Count positive mask pixels within a bbox region of the proto-space mask,
 * then normalise to original image area.
 *
 * The prototype mask is 256×256 (model input / 4). The bbox coords are in
 * the 1024×1024 model-input space, so we scale down by 4 to index into the
 * prototype mask.
 */
function maskAreaInBBox(
  mask: Float32Array,
  x1: number, y1: number, x2: number, y2: number,
  origW: number, origH: number,
  threshold = 0.5,
): number {
  const scale = PROTO_SIZE / 1024;
  const mx1 = Math.max(0,            Math.floor(x1 * scale));
  const my1 = Math.max(0,            Math.floor(y1 * scale));
  const mx2 = Math.min(PROTO_SIZE,   Math.ceil (x2 * scale));
  const my2 = Math.min(PROTO_SIZE,   Math.ceil (y2 * scale));

  let positive = 0;
  for (let row = my1; row < my2; row++) {
    for (let col = mx1; col < mx2; col++) {
      if (mask[row * PROTO_SIZE + col] > threshold) positive++;
    }
  }

  // Pixels in proto space → original image pixels → normalised area
  const protoPixelArea = (1024 / PROTO_SIZE) * (1024 / PROTO_SIZE);
  return (positive * protoPixelArea) / (origW * origH);
}

// ── Public interface ──────────────────────────────────────────────────────────

/**
 * Parse the two ONNX output tensors from FastSAM-s into RawDetections.
 *
 * @param output0Data  Flat data from output0 tensor [1, 37, 8400]
 * @param output1Data  Flat data from output1 tensor [1, 32, 256, 256]
 * @param scale        From letterboxToTensor()
 * @param padX         From letterboxToTensor()
 * @param padY         From letterboxToTensor()
 * @param origW        Original image width in px
 * @param origH        Original image height in px
 */
export function decodeDetections(
  output0Data: Float32Array,
  output1Data: Float32Array,
  scale: number,
  padX: number,
  padY: number,
  origW: number,
  origH: number,
): RawDetection[] {
  const NUM_ANCHORS = 8400;
  const NUM_ATTRS   = 37; // 4 box + 1 conf + 32 coeffs

  // ── Step 1: collect proposals above confidence threshold ──────────────────
  const proposals: Proposal[] = [];

  for (let a = 0; a < NUM_ANCHORS; a++) {
    // output0 is [1, 37, 8400] — iterate over the anchor dimension
    const conf = output0Data[4 * NUM_ANCHORS + a];
    if (conf < CONF_THRESHOLD) continue;

    const cx = output0Data[0 * NUM_ANCHORS + a];
    const cy = output0Data[1 * NUM_ANCHORS + a];
    const w  = output0Data[2 * NUM_ANCHORS + a];
    const h  = output0Data[3 * NUM_ANCHORS + a];

    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    const coeffs = new Float32Array(NUM_PROTOS);
    for (let k = 0; k < NUM_PROTOS; k++) {
      coeffs[k] = output0Data[(5 + k) * NUM_ANCHORS + a];
    }

    proposals.push({ x1, y1, x2, y2, conf, coeffs });
  }

  if (proposals.length === 0) return [];

  // ── Step 2: NMS ───────────────────────────────────────────────────────────
  const kept = nms(proposals, IOU_THRESHOLD);

  // ── Step 3: decode masks + compute areas ─────────────────────────────────
  // output1 is [1, 32, 256, 256] — drop the batch dimension
  const protos = output1Data.subarray(0);

  const results: RawDetection[] = [];

  for (const p of kept) {
    const mask = decodeMask(p.coeffs, protos);
    const area = maskAreaInBBox(mask, p.x1, p.y1, p.x2, p.y2, origW, origH);

    const bbox = modelBoxToNormalisedBBox(
      p.x1, p.y1, p.x2, p.y2,
      scale, padX, padY,
      origW, origH,
    );

    results.push({ bbox, area });
  }

  // Sort largest-first, matching server-side behaviour
  return results.sort((a, b) => b.area - a.area);
}