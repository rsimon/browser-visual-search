import { describe, it, expect, beforeAll } from 'vitest';
import { createEmbedder } from '../src/embed.js';
import { EMBEDDER_URL, fetchFixture, FIXTURE_IMAGES } from './setup.js';

const EMBEDDING_DIM = 512;

function norm(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

describe('embedder module', () => {
  let embedder: ReturnType<typeof createEmbedder>;
  let fileA: File;
  let fileB: File;

  beforeAll(async () => {
    embedder = createEmbedder({ modelUrl: EMBEDDER_URL, executionProviders: ['webgl','wasm'] });
    fileA = await fetchFixture(FIXTURE_IMAGES[0]);
    fileB = await fetchFixture(FIXTURE_IMAGES[1]);
  });

  it('returns a 512-dim Float32Array for a full image', async () => {
    const emb = await embedder.embed(fileA);
    expect(emb).toBeInstanceOf(Float32Array);
    expect(emb.length).toBe(EMBEDDING_DIM);
  });

  it('output is unit-normalised', async () => {
    const emb = await embedder.embed(fileA);
    expect(norm(emb)).toBeCloseTo(1.0, 4);
  });

  it('region bounding box works', async () => {
    const emb = await embedder.embed(fileA, [0.25, 0.25, 0.5, 0.5]);
    expect(emb.length).toBe(EMBEDDING_DIM);
    expect(norm(emb)).toBeCloseTo(1.0, 4);
  });

  it('full and region embeddings differ', async () => {
    const full = await embedder.embed(fileA);
    const region = await embedder.embed(fileA, [0, 0, 0.5, 0.5]);
    expect(cosineSim(full, region)).toBeLessThan(0.999);
  });

  it('two different images produce different embeddings', async () => {
    const embA = await embedder.embed(fileA);
    const embB = await embedder.embed(fileB);
    expect(cosineSim(embA, embB)).toBeLessThan(0.999);
  });

  it('embedding is deterministic', async () => {
    const e1 = await embedder.embed(fileA);
    const e2 = await embedder.embed(fileA);
    expect(cosineSim(e1, e2)).toBeCloseTo(1.0, 5);
  });

  describe('worker mode', () => {
    it('can initialise a worker embedder and compute an embedding', async () => {
      const wembed = createEmbedder({ modelUrl: EMBEDDER_URL, executionProviders: ['webgl','wasm'], useWorker: true });
      const emb = await wembed.embed(fileA);
      expect(emb.length).toBe(EMBEDDING_DIM);
    });
  });
});