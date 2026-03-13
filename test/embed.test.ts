import { describe, it, expect, beforeAll } from 'vitest';
import { embedImage } from '../src/embedding/embed.js';
import { fetchFixture, FIXTURE_IMAGES, EMBEDDER_URL } from './setup.js';

const EMBEDDING_DIM = 512;

const norm = (v: Float32Array): number => {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

const cosineSim = (a: Float32Array, b: Float32Array): number => {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both unit-normalised, so dot = cosine similarity
}

describe('embedImage', () => {
  let fileA: File;
  let fileB: File;

  beforeAll(async () => {
    fileA = await fetchFixture(FIXTURE_IMAGES[0]);
    fileB = await fetchFixture(FIXTURE_IMAGES[1]);
  });

  it('returns a 512-dim Float32Array for a full image', async () => {
    const emb = await embedImage(fileA, undefined, { embedderUrl: EMBEDDER_URL });
    expect(emb).toBeInstanceOf(Float32Array);
    expect(emb.length).toBe(EMBEDDING_DIM);
  });

  it('output is unit-normalised (norm ≈ 1.0)', async () => {
    const emb = await embedImage(fileA, undefined, { embedderUrl: EMBEDDER_URL });
    expect(norm(emb)).toBeCloseTo(1.0, 4);
  });

  it('returns a valid embedding for a sub-region bbox', async () => {
    const emb = await embedImage(fileA, [0.25, 0.25, 0.5, 0.5], { embedderUrl: EMBEDDER_URL });
    expect(emb).toBeInstanceOf(Float32Array);
    expect(emb.length).toBe(EMBEDDING_DIM);
    expect(norm(emb)).toBeCloseTo(1.0, 4);
  });

  it('full-image embed and region embed of the same image differ', async () => {
    const full   = await embedImage(fileA, undefined, { embedderUrl: EMBEDDER_URL });
    const region = await embedImage(fileA, [0.0, 0.0, 0.5, 0.5], { embedderUrl: EMBEDDER_URL });
    expect(cosineSim(full, region)).toBeLessThan(0.999);
  });

  it('two different images produce different embeddings', async () => {
    const embA = await embedImage(fileA, undefined, { embedderUrl: EMBEDDER_URL });
    const embB = await embedImage(fileB, undefined, { embedderUrl: EMBEDDER_URL });
    expect(cosineSim(embA, embB)).toBeLessThan(0.999);
  });

  it('embedding is deterministic — same input produces same output', async () => {
    const emb1 = await embedImage(fileA, undefined, { embedderUrl: EMBEDDER_URL });
    const emb2 = await embedImage(fileA, undefined, { embedderUrl: EMBEDDER_URL });
    expect(cosineSim(emb1, emb2)).toBeCloseTo(1.0, 5);
  });
});