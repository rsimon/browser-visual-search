import { createEmbedder } from './embed.js';
import type { BBox } from './types.js';

// Worker-side proxy for embedding. Accepts messages from the main thread and
// responds with embeddings. File objects are transferred via postMessage and
// processed in the worker context.

interface InitMessage {
  type: 'init';
  modelUrl: string;
  executionProviders?: string[];
}

interface EmbedMessage {
  type: 'embed';
  id: number;
  file: File;
  bbox?: BBox;
}

interface EmbedBatchMessage {
  type: 'embedBatch';
  id: number;
  file: File;
  bboxes: Array<BBox | null>;
  batchSize?: number;
}

interface ResultMessage {
  type: 'result';
  id: number;
  embedding: Float32Array;
}

interface BatchResultMessage {
  type: 'batchResult';
  id: number;
  embeddings: Float32Array[];
}

interface ErrorMessage {
  type: 'error';
  id: number;
  message: string;
}

let embedder: ReturnType<typeof createEmbedder> | null = null;

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as InitMessage | EmbedMessage | EmbedBatchMessage;
  try {
    if (msg.type === 'init') {
      embedder = createEmbedder({
        modelUrl: msg.modelUrl,
        executionProviders: msg.executionProviders,
      });
      self.postMessage({ type: 'inited' });
    } else if (msg.type === 'embed') {
      if (!embedder) throw new Error('Embedder not initialised');
      const emb = await embedder.embed(msg.file, msg.bbox);
      self.postMessage({ type: 'result', id: msg.id, embedding: emb }, [emb.buffer]);
    } else if (msg.type === 'embedBatch') {
      if (!embedder) throw new Error('Embedder not initialised');
      const embs = await embedder.embedBatch(msg.file, msg.bboxes, { batchSize: msg.batchSize });
      const transferList = embs.map(e => e.buffer);
      self.postMessage({ type: 'batchResult', id: msg.id, embeddings: embs }, transferList);
    }
  } catch (err: any) {
    const id = (msg as any).id ?? -1;
    self.postMessage({ type: 'error', id, message: err?.message ?? String(err) });
  }
};
