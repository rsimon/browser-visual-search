/**
 * indexImage.test.ts
 * ------------------
 * Validates the FastSAM segmentation + CLIP embedding pipeline end-to-end.
 *
 * Checks:
 *   - At least one segment is returned
 *   - Every segment has a valid normalised bbox and area
 *   - Every segment has a 512-dim unit-normalised embedding
 *   - Segments are sorted largest-area-first (matching server-side behaviour)
 *   - Results are deterministic
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { getVS, fetchFixture, assertValidBBox, FIXTURE_IMAGES } from './setup.js';
import type { VisualSearch } from '../src/visual-search.js';
import type { Segment } from '../src/types.js';

const EMBEDDING_DIM = 512;

function norm(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe.skip('indexImage', () => {
  let vs: VisualSearch;
  let fileA: File;
  let segments: Segment[];

  beforeAll(async () => {
    vs       = await getVS();
    fileA    = await fetchFixture(FIXTURE_IMAGES[0]);
    segments = await vs.indexImage({ id: 'test-image-a', file: fileA });
  }, 120000);

  it('returns at least one segment', () => {
    expect(segments.length).toBeGreaterThan(0);
  });

  it('logs segment count (informational)', () => {
    console.log(`indexImage: ${segments.length} segments for ${FIXTURE_IMAGES[0]}`);
    // Not a real assertion — just useful to see in test output
  });

  it('every segment has a valid normalised bbox', () => {
    for (const seg of segments) {
      assertValidBBox(seg.bbox, `segment bbox`);
    }
  });

  it('every segment has a normalised area in (0, 1]', () => {
    for (const seg of segments) {
      expect(seg.area).toBeGreaterThan(0);
      expect(seg.area).toBeLessThanOrEqual(1.0);
    }
  });

  it('every segment has a 512-dim unit-normalised embedding', () => {
    for (const seg of segments) {
      expect(seg.embedding).toBeInstanceOf(Float32Array);
      expect(seg.embedding.length).toBe(EMBEDDING_DIM);
      expect(norm(seg.embedding)).toBeCloseTo(1.0, 4);
    }
  });

  it('segments are sorted largest-area-first', () => {
    for (let i = 1; i < segments.length; i++) {
      expect(segments[i - 1].area).toBeGreaterThanOrEqual(segments[i].area);
    }
  });

  it('results are deterministic — same image produces same segment count', async () => {
    const segments2 = await vs.indexImage({ id: 'test-image-a-2', file: fileA });
    expect(segments2.length).toBe(segments.length);
  });
});