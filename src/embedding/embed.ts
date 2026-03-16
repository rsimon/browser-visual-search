import * as ort from 'onnxruntime-web';
import type { BBox } from '../types.js';

const EMBEDDING_DIM = 512;
const CLIP_INPUT_SIZE = 224;

let embedderSession: ort.InferenceSession | null = null;

const loadEmbedder = async (url: string, providers: string[] = ['webgpu', 'wasm']): Promise<ort.InferenceSession> => {
  if (!embedderSession)
    embedderSession = await ort.InferenceSession.create(url, { executionProviders: providers });

  return embedderSession;
}

const cropToClipTensor = (bitmap: ImageBitmap, bbox?: BBox): Float32Array => {
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

const l2Normalise = (vec: Float32Array): Float32Array => {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

export const embedImage = async (
  file: File,
  bbox?: BBox,
  options: { embedderUrl: string; executionProviders?: string[] } = { embedderUrl: '' }
): Promise<Float32Array> => {
  if (!options.embedderUrl) throw new Error('embedderUrl is required');

  const bitmap = await createImageBitmap(file);
  const embedder = await loadEmbedder(options.embedderUrl, options.executionProviders);

  const tensor = cropToClipTensor(bitmap, bbox);
  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE]);
  const feeds = { [embedder.inputNames[0]]: inputTensor };
  const output = await embedder.run(feeds);
  const raw = output[embedder.outputNames[0]].data as Float32Array;

  bitmap.close();
  return l2Normalise(new Float32Array(raw));
}

export const embedBatch = async (
  bitmap: ImageBitmap,
  bboxes: Array<BBox | undefined>,
  options: { embedderUrl: string; executionProviders?: string[] },
  batchSize = 32,
): Promise<Float32Array[]> => {
  const embedder = await loadEmbedder(options.embedderUrl, options.executionProviders);
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