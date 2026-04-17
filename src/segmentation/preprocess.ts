/**
 * Image pre-processing for FastSAM inference.
 *
 * FastSAM-s expects a [1, 3, 1280, 1280] float32 tensor.
 * Pixel values normalised to [0, 1], RGB channel order.
 * The image is letterboxed (aspect ratio preserved, padded with 0.5 grey).
 */
export const MODEL_INPUT_SIZE = 1280;

export interface LetterboxResult {

  /** [1, 3, H, W] flat, row-major */
  tensor: Float32Array;      

  /** Scale factor applied (same for x and y — uniform scaling). */
  scale: number;

  /** Padding applied to left/right (px, in model-input space). */
  padX: number;

  /** Padding applied to top/bottom (px, in model-input space). */
  padY: number;

}

/**
 * Letterbox an ImageBitmap to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE,
 * then return a normalised float32 tensor and the transform parameters
 * needed to map detections back to the original image coordinate space.
 */
export const letterboxToTensor = (bitmap: ImageBitmap): LetterboxResult => {
  const S = MODEL_INPUT_SIZE;
  const { width: srcW, height: srcH } = bitmap;

  // Uniform scale so the longest side fits in S
  const scale = Math.min(S / srcW, S / srcH);
  const scaledW = Math.round(srcW * scale);
  const scaledH = Math.round(srcH * scale);

  // Padding to centre the scaled image
  const padX = Math.floor((S - scaledW) / 2);
  const padY = Math.floor((S - scaledH) / 2);

  // Draw into an off-screen canvas
  const canvas = new OffscreenCanvas(S, S);
  const ctx = canvas.getContext('2d')!;

  // Fill with mid-grey (matching Ultralytics letterbox default: 114/255 ≈ 0.447)
  ctx.fillStyle = `rgb(114,114,114)`;
  ctx.fillRect(0, 0, S, S);
  ctx.drawImage(bitmap, padX, padY, scaledW, scaledH);

  const imageData = ctx.getImageData(0, 0, S, S);
  const { data } = imageData; // Uint8ClampedArray, RGBA interleaved

  // Convert RGBA interleaved → RGB planar, normalised to [0,1]
  const tensor = new Float32Array(3 * S * S);
  const planeSize = S * S;

  for (let i = 0; i < planeSize; i++) {
    tensor[i]                   = data[i * 4]     / 255; // R
    tensor[planeSize + i]       = data[i * 4 + 1] / 255; // G
    tensor[planeSize * 2 + i]   = data[i * 4 + 2] / 255; // B
  }

  return { tensor, scale, padX, padY };
}

/**
 * Map a model-space bbox [x1, y1, x2, y2] (absolute pixels in the
 * letterboxed 1024×1024 space) back to normalised [x, y, w, h]
 * coordinates in the original image space.
 */
export const modelBoxToBBox = (
  x1: number, y1: number, x2: number, y2: number,
  scale: number, padX: number, padY: number,
  origW: number, origH: number,
): { normalizedBounds: [number, number, number, number], pxBounds: [number, number, number, number] } => {
  // Remove padding, undo scale
  const ox1 = Math.max(0, (x1 - padX) / scale);
  const oy1 = Math.max(0, (y1 - padY) / scale);
  const ox2 = Math.min(origW, (x2 - padX) / scale);
  const oy2 = Math.min(origH, (y2 - padY) / scale);

  return {
    normalizedBounds: [
      ox1 / origW,
      oy1 / origH,
      (ox2 - ox1) / origW,
      (oy2 - oy1) / origH,
    ],
    pxBounds: [
      Math.round(ox1), 
      Math.round(oy1),
      Math.round(ox2 - ox1),
      Math.round(oy2 - oy1)
    ]
  }
}