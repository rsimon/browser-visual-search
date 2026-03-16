import { beforeAll, describe, expect, it } from 'vitest';
import { createIndex, EMBEDDING_DIM, type LoadIndexOptions } from '../src/search/load-index.js';
import { embedBatch } from '../src/embedding/embed.js';
import type { IndexedImage, VisualSearchIndex } from '../src/types.js';
import { assertInRange, assertValidBBox, EMBEDDER_URL, fetchFixture, FIXTURE_IMAGES } from './setup.js';

const OPTS: LoadIndexOptions = { embedderUrl: EMBEDDER_URL };

// ── Build a real index fixture ─────────────────────────────────────────────

const SEGMENTS = [
  { bbox: [0, 0, 1, 1]       as [number,number,number,number], area: 1    },
  { bbox: [0, 0, 0.5, 0.5]   as [number,number,number,number], area: 0.25 },
  { bbox: [0.5, 0.5, 0.5, 0.5] as [number,number,number,number], area: 0.25 },
];

let index: VisualSearchIndex;
let fileA: File;
let fileB: File;
let fileC: File;

beforeAll(async () => {
  [fileA, fileB, fileC] = await Promise.all(FIXTURE_IMAGES.map(fetchFixture));

  const files = [
    { id: 'image-a.jpg', file: fileA },
    { id: 'image-b.jpg', file: fileB },
    { id: 'image-c.jpg', file: fileC },
  ];

  const images: IndexedImage[] = [];
  const allEmbeddings: Float32Array[] = [];
  let nextRow = 0;

  for (const { id, file } of files) {
    const bitmap = await createImageBitmap(file);
    const embeddings = await embedBatch(bitmap, SEGMENTS.map(s => s.bbox), OPTS);
    bitmap.close();

    images.push({
      imageId:   id,
      indexedAt: new Date().toISOString(),
      segments:  SEGMENTS.map((s, j) => ({
        bbox:         s.bbox,
        area:         s.area,
        embeddingRow: nextRow++,
      })),
    });

    allEmbeddings.push(...embeddings);
  }

  const flat = new Float32Array(allEmbeddings.length * EMBEDDING_DIM);
  allEmbeddings.forEach((vec, i) => flat.set(vec, i * EMBEDDING_DIM));

  index = createIndex(images, flat, {} as FileSystemDirectoryHandle, OPTS);
});

// ── getImage ───────────────────────────────────────────────────────────────

describe('getImage', () => {
  it('returns the correct image for a known id', () => {
    const img = index.getImage('image-a.jpg');
    expect(img).toBeDefined();
    expect(img!.imageId).toBe('image-a.jpg');
    expect(img!.segments).toHaveLength(SEGMENTS.length);
  });

  it('returns undefined for an unknown id', () => {
    expect(index.getImage('does-not-exist.jpg')).toBeUndefined();
  });
});

// ── query ──────────────────────────────────────────────────────────────────

describe('query', () => {
  it('returns results for every image in the index', async () => {
    const results = await index.query(fileA);
    const ids = new Set(results.map(r => r.imageId));
    expect(ids.has('image-a.jpg')).toBe(true);
    expect(ids.has('image-b.jpg')).toBe(true);
    expect(ids.has('image-c.jpg')).toBe(true);
  });

  it('scores are in [-1, 1] and results are sorted descending', async () => {
    const results = await index.query(fileA);
    for (const r of results)
      assertInRange(r.score, -1, 1, 'score');
    for (let i = 1; i < results.length; i++)
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
  });

  it('top result for a full-image query matches the queried image', async () => {
    for (const [id, file] of [['image-a.jpg', fileA], ['image-b.jpg', fileB], ['image-c.jpg', fileC]] as const) {
      const results = await index.query(file);
      expect(results[0].imageId).toBe(id);
    }
  });

  it('bbox query returns valid results', async () => {
    const results = await index.query(fileA, [0.25, 0.25, 0.5, 0.5]);
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      assertInRange(r.score, -1, 1, 'score');
      assertValidBBox(r.bbox);
      expect(typeof r.imageId).toBe('string');
    }
  });

  it('respects topK', async () => {
    const results = await index.query(fileA, undefined, { topK: 2 });
    expect(results).toHaveLength(2);
  });

  it('different images produce different top results', async () => {
    const [rA, rB, rC] = await Promise.all([
      index.query(fileA),
      index.query(fileB),
      index.query(fileC),
    ]);
    expect(rA[0].imageId).not.toBe(rB[0].imageId);
    expect(rA[0].imageId).not.toBe(rC[0].imageId);
  });
});