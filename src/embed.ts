/**
 * embed.ts
 * --------
 * CLIP embedding utilities. Exports a factory that produces an `Embedder`
 * capable of embedding full images or regions.  Embeddings are l2-normalised
 * automatically. The heavy ONNX dependency is loaded lazily on first use.
 *
 * An optional Web Worker proxy will be added later; the constructor accepts
 * `useWorker` and `worker` options.
 */

import type { BBox } from './types.js';

export interface EmbedderOptions {
  /** URL to an ONNX model to load. */
  modelUrl?: string;

  /** Pre‑created ORT session; if provided, it takes precedence. */
  session?: any;

  /** Preferred execution providers (e.g. ['webgpu','wasm']). */
  executionProviders?: string[];

  /** If true, the embedder will run in a Web Worker (worker code upcoming). */
  useWorker?: boolean;

  /** A pre‑existing Worker instance to proxy; optional. */
  worker?: Worker;
}

export interface Embedder {
  embed(file: File, bbox?: BBox, options?: { signal?: AbortSignal }): Promise<Float32Array>;
  embedBitmap(bitmap: ImageBitmap, bbox?: BBox | null): Promise<Float32Array>;
  embedBatch(
    fileOrBitmap: File | ImageBitmap,
    bboxes: Array<BBox | null>,
    options?: { signal?: AbortSignal; batchSize?: number },
  ): Promise<Float32Array[]>;
}

function cropToClipTensor(
  bitmap: ImageBitmap,
  bbox: BBox | null,
): Float32Array {
  const { width: W, height: H } = bitmap;

  let sx = 0, sy = 0, sw = W, sh = H;
  if (bbox) {
    sx = Math.round(bbox[0] * W);
    sy = Math.round(bbox[1] * H);
    sw = Math.max(1, Math.round(bbox[2] * W));
    sh = Math.max(1, Math.round(bbox[3] * H));
  }

  const S = 224;
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

function l2Normalise(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

function createWorkerProxy(worker: Worker): Embedder {
  let nextId = 1;
  const pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();

  worker.onmessage = (ev: MessageEvent) => {
    const msg = ev.data;
    if (msg.type === 'result') {
      pending.get(msg.id)?.resolve(msg.embedding);
      pending.delete(msg.id);
    } else if (msg.type === 'batchResult') {
      pending.get(msg.id)?.resolve(msg.embeddings);
      pending.delete(msg.id);
    } else if (msg.type === 'error') {
      pending.get(msg.id)?.reject(new Error(msg.message));
      pending.delete(msg.id);
    }
  };

  return {
    async embed(file: File, bbox?: BBox, options?: { signal?: AbortSignal }) {
      options?.signal?.throwIfAborted?.();
      const id = nextId++;
      const promise = new Promise<Float32Array>((res, rej) => pending.set(id, { resolve: res, reject: rej }));
      worker.postMessage({ type: 'embed', id, file, bbox });
      return promise;
    },

    embedBitmap() {
      return Promise.reject(new Error('embedBitmap not supported in worker; pass a File'));
    },

    async embedBatch(
      fileOrBitmap: File | ImageBitmap,
      bboxes: Array<BBox | null>,
      options?: { signal?: AbortSignal; batchSize?: number },
    ): Promise<Float32Array[]> {
      options?.signal?.throwIfAborted?.();
      if (!(fileOrBitmap instanceof File)) {
        return Promise.reject(new Error('embedBatch with ImageBitmap not supported in worker'));
      }
      const id = nextId++;
      const promise = new Promise<Float32Array[]>((res, rej) => pending.set(id, { resolve: res, reject: rej }));
      worker.postMessage({ type: 'embedBatch', id, file: fileOrBitmap, bboxes, batchSize: options?.batchSize });
      return promise;
    },
  };
}

export function createEmbedder(opts: EmbedderOptions = {}): Embedder {
  if (opts.useWorker || opts.worker) {
    const worker = opts.worker ?? new Worker(new URL('./embed.worker.js', import.meta.url), { type: 'module' });
    return createWorkerProxy(worker);
  }

  let _session: any | null = null;

  async function getSession() {
    if (_session) return _session;
    const ort = await import('onnxruntime-web');
    if (opts.session) {
      _session = opts.session;
    } else {
      if (!opts.modelUrl) {
        throw new Error('Embedder requires a modelUrl or a pre‑created session');
      }
      _session = await ort.InferenceSession.create(opts.modelUrl, {
        executionProviders: opts.executionProviders ?? ['wasm'],
      });
    }
    return _session;
  }

  async function embedBitmap(bitmap: ImageBitmap, bbox: BBox | null = null): Promise<Float32Array> {
    const session = await getSession();
    const tensor = cropToClipTensor(bitmap, bbox);
    const ort = await import('onnxruntime-web');
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 224, 224]);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const output = await session.run(feeds);
    const raw = output[session.outputNames[0]].data as Float32Array;
    return l2Normalise(new Float32Array(raw));
  }

  async function embed(file: File, bbox?: BBox, options?: { signal?: AbortSignal }): Promise<Float32Array> {
    options?.signal?.throwIfAborted?.();
    const bitmap = await createImageBitmap(file);
    try {
      return await embedBitmap(bitmap, bbox ?? null);
    } finally {
      bitmap.close();
    }
  }

  async function embedBatch(
    fileOrBitmap: File | ImageBitmap,
    bboxes: Array<BBox | null>,
    options: { signal?: AbortSignal; batchSize?: number } = {},
  ): Promise<Float32Array[]> {
    let bitmap: ImageBitmap;
    if (fileOrBitmap instanceof File) {
      bitmap = await createImageBitmap(fileOrBitmap);
    } else {
      bitmap = fileOrBitmap;
    }
    try {
      const session = await getSession();
      const results: Float32Array[] = [];
      const batchSize = options.batchSize ?? 32;

      for (let i = 0; i < bboxes.length; i += batchSize) {
        options.signal?.throwIfAborted?.();
        const batchBBoxes = bboxes.slice(i, i + batchSize);
        const tensors = batchBBoxes.map(bbox => cropToClipTensor(bitmap, bbox));

        const B = tensors.length;
        const flat = new Float32Array(B * 3 * 224 * 224);
        tensors.forEach((t, j) => flat.set(t, j * t.length));

        const ort = await import('onnxruntime-web');
        const inputTensor = new ort.Tensor('float32', flat, [B, 3, 224, 224]);
        const feeds = { [session.inputNames[0]]: inputTensor };
        const output = await session.run(feeds);
        const raw = output[session.outputNames[0]].data as Float32Array;

        for (let j = 0; j < B; j++) {
          const vec = new Float32Array(512);
          for (let k = 0; k < 512; k++) {
            vec[k] = raw[j * 512 + k];
          }
          results.push(l2Normalise(vec));
        }
      }
      return results;
    } finally {
      if (fileOrBitmap instanceof File) {
        bitmap.close();
      }
    }
  }

  return { embed, embedBitmap, embedBatch };
}
