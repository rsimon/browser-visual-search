/**
 * embedRegion.test.ts
 * -------------------
 * Validates the CLIP embedding pipeline end-to-end.
 *
 * Checks:
 *   - Output is a 512-dim Float32Array
 *   - Vector is unit-normalised (norm ≈ 1.0)
 *   - Full-image embed and cropped-region embed both work
 *   - Two different images produce meaningfully different embeddings
 *   - The same image embedded twice produces identical results (determinism)
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { getVS, fetchFixture, FIXTURE_IMAGES } from './setup.js';
import type { VisualSearch } from '../src/visual-search.js';

const EMBEDDING_DIM = 512;

function norm(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both unit-normalised, so dot = cosine similarity
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('embedRegion', () => {
  let vs: VisualSearch;
  let fileA: File;
  let fileB: File;

  beforeAll(async () => {
    vs    = await getVS();
    fileA = await fetchFixture(FIXTURE_IMAGES[0]);
    fileB = await fetchFixture(FIXTURE_IMAGES[1]);
  });

  it('returns a 512-dim Float32Array for a full image', async () => {
    const emb = await vs.embedRegion(fileA);
    expect(emb).toBeInstanceOf(Float32Array);
    expect(emb.length).toBe(EMBEDDING_DIM);
  });

  it('output is unit-normalised (norm ≈ 1.0)', async () => {
    const emb = await vs.embedRegion(fileA);
    expect(norm(emb)).toBeCloseTo(1.0, 4);
  });

  it('returns a valid embedding for a sub-region bbox', async () => {
    // Embed the centre quarter of the image
    const emb = await vs.embedRegion(fileA, [0.25, 0.25, 0.5, 0.5]);
    expect(emb).toBeInstanceOf(Float32Array);
    expect(emb.length).toBe(EMBEDDING_DIM);
    expect(norm(emb)).toBeCloseTo(1.0, 4);
  });

  it('full-image embed and region embed of the same image differ', async () => {
    const full   = await vs.embedRegion(fileA);
    const region = await vs.embedRegion(fileA, [0.0, 0.0, 0.5, 0.5]);
    // They should not be identical
    expect(cosineSim(full, region)).toBeLessThan(0.999);
  });

  it('two different images produce different embeddings', async () => {
    const embA = await vs.embedRegion(fileA);
    const embB = await vs.embedRegion(fileB);
    expect(cosineSim(embA, embB)).toBeLessThan(0.999);
  });

  it('embedding is deterministic — same input produces same output', async () => {
    const emb1 = await vs.embedRegion(fileA);
    const emb2 = await vs.embedRegion(fileA);
    expect(cosineSim(emb1, emb2)).toBeCloseTo(1.0, 5);
  });
});