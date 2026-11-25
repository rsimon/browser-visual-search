import { beforeEach, describe, expect, it } from 'vitest';
import { createIndex, openIndex, EMBEDDING_DIM } from '../src/search-index/index.js';
import type { IndexedImage } from '../src/types.js';

// A synthetic File System Access API fixture: no model downloads or inference.
class MemoryDirectory {
  files = new Map<string, Uint8Array>();
  writes = 0;

  handle: FileSystemDirectoryHandle = {
    getDirectoryHandle: async () => this.handle,
    getFileHandle: async (name: string, options?: FileSystemGetFileOptions) => {
      if (!this.files.has(name) && !options?.create)
        throw new DOMException('Missing file', 'NotFoundError');
      return {
        getFile: async () => new File([this.files.get(name)!.slice().buffer as ArrayBuffer], name),
        createWritable: async () => ({
          write: async (data: Uint8Array | ArrayBuffer) => {
            this.writes++;
            this.files.set(name, data instanceof Uint8Array ? data.slice() : new Uint8Array(data.slice(0)));
          },
          close: async () => {},
        }),
      };
    },
  } as unknown as FileSystemDirectoryHandle;

  manifest() {
    return JSON.parse(new TextDecoder().decode(this.files.get('index.json')));
  }

  setManifest(value: unknown) {
    this.files.set('index.json', new TextEncoder().encode(JSON.stringify(value)));
  }

  snapshot() {
    return [...this.files].map(([name, bytes]) => [name, [...bytes]]);
  }
}

const opts = { embedderUrl: '/models/clip-old.onnx', segmenterUrl: '/models/sam-old.onnx' };
let directory: MemoryDirectory;

beforeEach(async () => {
  directory = new MemoryDirectory();
  const images: IndexedImage[] = [{
    imageId: 'synthetic.jpg',
    indexedAt: '2026-01-01T00:00:00Z',
    segments: [{ normalizedBounds: [0, 0, 1, 1], pxBounds: [0, 0, 1, 1], area: 1, embeddingRow: 0 }],
  }];
  const vector = new Float32Array(EMBEDDING_DIM);
  vector[0] = 1;
  await createIndex(images, [vector], directory.handle, opts).save();
});

const rejectsWithoutWriting = async (options: Parameters<typeof openIndex>[1], pattern: RegExp) => {
  const before = directory.snapshot();
  const writes = directory.writes;
  await expect(openIndex(directory.handle, options)).rejects.toThrow(pattern);
  expect(directory.snapshot()).toEqual(before);
  expect(directory.writes).toBe(writes);
};

describe('persisted model compatibility', () => {
  it('opens matching build models and keeps the original vectors', async () => {
    const loaded = await openIndex(directory.handle, opts);
    expect(loaded.embeddings[0][0]).toBe(1);
    expect(loaded.images[0].imageId).toBe('synthetic.jpg');
  });

  it('keeps segmenter metadata when a query-only index is saved', async () => {
    const before = directory.snapshot();
    const loaded = await openIndex(directory.handle, { embedderUrl: opts.embedderUrl });
    await loaded.save();
    expect(directory.manifest().models).toEqual({ embedder: 'clip-old', segmenter: 'sam-old' });
    expect([...directory.files.get('embeddings.bin')!]).toEqual(before[1][1]);
  });

  it('rejects an incompatible embedder even with create enabled', async () => {
    await rejectsWithoutWriting({ ...opts, embedderUrl: '/models/clip-new.onnx', create: true }, /embedder.*mismatch/i);
  });

  it('rejects an incompatible segmenter when adding images is enabled', async () => {
    await rejectsWithoutWriting({ ...opts, segmenterUrl: '/models/sam-new.onnx' }, /segmenter.*mismatch/i);
  });

  it('uses the existing model-name convention rather than introducing URL identity', async () => {
    const loaded = await openIndex(directory.handle, { embedderUrl: 'https://example.invalid/clip-old.onnx' });
    expect(loaded.embeddings).toHaveLength(1);
  });

  it('rejects a legacy index with no model metadata', async () => {
    const manifest = directory.manifest();
    delete manifest.models;
    manifest.model = 'clip-old';
    directory.setManifest(manifest);
    await rejectsWithoutWriting(opts, /model metadata.*rebuild/i);
  });

  it.each([null, {}, { embedder: '', segmenter: 'sam-old' },
    { embedder: 1, segmenter: 'sam-old' }, { embedder: 'clip-old', segmenter: 1 }])(
    'rejects malformed model metadata: %j', async models => {
      directory.setManifest({ ...directory.manifest(), models });
      await rejectsWithoutWriting(opts, /model metadata.*rebuild/i);
    });

  it('allows querying an index with an explicitly unknown segmenter but preserves that fact', async () => {
    directory.setManifest({ ...directory.manifest(), models: { embedder: 'clip-old', segmenter: '' } });
    const loaded = await openIndex(directory.handle, { embedderUrl: opts.embedderUrl });
    await loaded.save();
    expect(directory.manifest().models.segmenter).toBe('');
    await rejectsWithoutWriting(opts, /segmenter.*mismatch/i);
    await rejectsWithoutWriting({ ...opts, segmenterUrl: '' }, /segmenter.*mismatch/i);
  });

  it('still rejects a corrupt embeddings length without writing', async () => {
    directory.files.set('embeddings.bin', new Uint8Array(0));
    await rejectsWithoutWriting(opts, /length mismatch/);
  });

  it('still creates a new empty index when no saved index exists', async () => {
    directory = new MemoryDirectory();
    const created = await openIndex(directory.handle, { ...opts, create: true });
    expect(created.images).toEqual([]);
    expect(directory.writes).toBe(0);
    await created.save();
    expect(directory.manifest().models).toEqual({ embedder: 'clip-old', segmenter: 'sam-old' });
  });
});
