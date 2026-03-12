import { describe, it, expect, beforeAll } from 'vitest';
import { createSegmenter } from '../src/segment.js';
import { SEGMENTER_URL, fetchFixture, FIXTURE_IMAGES, assertValidBBox } from './setup.js';

interface Seg {
  bbox: [number, number, number, number];
  area: number;
}

describe('segmenter module', () => {
  let segmenter: ReturnType<typeof createSegmenter>;
  let fileA: File;
  let fileB: File;

  beforeAll(async () => {
    segmenter = createSegmenter({ modelUrl: SEGMENTER_URL, executionProviders: ['webgl','wasm'] });
    fileA = await fetchFixture(FIXTURE_IMAGES[0]);
    fileB = await fetchFixture(FIXTURE_IMAGES[1]);
  });

  it('returns at least one segment', async () => {
    const segments = await segmenter.segment(fileA);
    expect(segments.length).toBeGreaterThan(0);
  });

  it('segments have valid bbox and area', async () => {
    const segments = await segmenter.segment(fileA);
    for (const seg of segments) {
      assertValidBBox(seg.bbox);
      expect(seg.area).toBeGreaterThanOrEqual(0); // some masks may be tiny or empty
      expect(seg.area).toBeLessThanOrEqual(1);
    }
  });

  it('different images may produce different counts', async () => {
    const segA = await segmenter.segment(fileA);
    const segB = await segmenter.segment(fileB);
    // we don't assert difference, just that both run without error
    expect(segA.length).toBeGreaterThan(0);
    expect(segB.length).toBeGreaterThan(0);
  });

  it('deterministic: calling twice yields same count', async () => {
    const s1 = await segmenter.segment(fileA);
    const s2 = await segmenter.segment(fileA);
    expect(s1.length).toBe(s2.length);
  });
});