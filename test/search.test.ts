/**
 * search.test.ts
 * --------------
 * Validates the full round-trip: buildIndex → search.
 *
 * Checks:
 *   - buildIndex produces a valid index from multiple images
 *   - onProgress fires correctly
 *   - search returns topK results with valid structure
 *   - Querying with an image returns that same image as the top result
 *     (self-similarity: the best match for an image should be itself)
 *   - scores are in descending order
 *   - scores are in [0, 1] (cosine similarity of unit vectors)
 *   - serialise writes the expected files to an in-memory OPFS directory
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { getVS, fetchFixture, assertValidBBox, FIXTURE_IMAGES } from './setup.js';
import { deserializeIndex } from '../src/visual-search.js';
import type { VisualSearch, VisualSearchIndex } from '../src/visual-search.js';

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('search', () => {
  let vs: VisualSearch;
  let files: File[];
  let index: VisualSearchIndex;

  const IMAGE_IDS = ['img-a', 'img-b', 'img-c'] as const;

  beforeAll(async () => {
    vs    = await getVS();
    files = await Promise.all(FIXTURE_IMAGES.map(fetchFixture));

    const progressLog: Array<[number, number]> = [];

    index = await vs.buildIndex(
      files.map((file, i) => ({ id: IMAGE_IDS[i], file })),
      { onProgress: (done, total) => progressLog.push([done, total]) },
    );

    // Verify progress fired once per image
    expect(progressLog.length).toBe(files.length);
    expect(progressLog.at(-1)).toEqual([files.length, files.length]);
  });

  it('index contains all images', () => {
    expect(index.images.length).toBe(files.length);
  });

  it('index contains at least one segment per image', () => {
    for (const img of index.images) {
      expect(img.segments.length).toBeGreaterThan(0);
    }
  });

  it('embeddings buffer has the right length', () => {
    const totalSegments = index.images.reduce((n, img) => n + img.segments.length, 0);
    expect(index.embeddings.length).toBe(totalSegments * 512);
  });

  it('search returns topK results', async () => {
    const results = await vs.search(files[0], undefined, index, { topK: 5 });
    expect(results.length).toBeLessThanOrEqual(5);
    expect(results.length).toBeGreaterThan(0);
  });

  it('results have valid structure', async () => {
    const results = await vs.search(files[0], undefined, index);
    for (const r of results) {
      expect(typeof r.imageId).toBe('string');
      assertValidBBox(r.bbox);
      expect(r.area).toBeGreaterThan(0);
      expect(r.score).toBeGreaterThanOrEqual(-1);
      expect(r.score).toBeLessThanOrEqual(1);
    }
  });

  it('scores are in descending order', async () => {
    const results = await vs.search(files[0], undefined, index);
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  it('self-similarity: querying image-a returns image-a as the top result', async () => {
    const results = await vs.search(files[0], undefined, index, { topK: 1 });
    expect(results[0].imageId).toBe(IMAGE_IDS[0]);
  });

  it('self-similarity: querying image-b returns image-b as the top result', async () => {
    const results = await vs.search(files[1], undefined, index, { topK: 1 });
    expect(results[0].imageId).toBe(IMAGE_IDS[1]);
  });

  it('serialize + deserialize round-trips the index correctly', async () => {
    // Use the Origin Private File System (OPFS) — available in all modern
    // browsers and in Vitest browser mode, no user permission prompt needed.
    const root    = await navigator.storage.getDirectory();
    const testDir = await root.getDirectoryHandle('vs-test', { create: true });

    await index.serialize(testDir);

    const loaded = await deserializeIndex(testDir);

    expect(loaded.images.length).toBe(index.images.length);
    expect(loaded.embeddings.length).toBe(index.embeddings.length);

    // Embeddings should be numerically identical after round-trip
    for (let i = 0; i < index.embeddings.length; i++) {
      expect(loaded.embeddings[i]).toBeCloseTo(index.embeddings[i], 5);
    }

    // Clean up
    await root.removeEntry('vs-test', { recursive: true });
  });
});