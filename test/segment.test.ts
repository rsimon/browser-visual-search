import { describe, it, expect, beforeAll } from 'vitest';
import { segmentImage } from '../src/segmentation/segment.js';
import { fetchFixture, assertValidBBox, FIXTURE_IMAGES, SEGMENTER_URL } from './setup.js';

describe('segmentImage', () => {
  let fileA: File;
  let segments: Array<{ normalizedBounds: [number, number, number, number]; area: number }>;

  beforeAll(async () => {
    fileA = await fetchFixture(FIXTURE_IMAGES[0]);
    segments = await segmentImage(fileA, { segmenterUrl: SEGMENTER_URL });
  });

  it('returns at least one segment', () => {
    expect(segments.length).toBeGreaterThan(0);
  });

  it('every segment has a valid normalised bbox', () => {
    for (const seg of segments) {
      assertValidBBox(seg.normalizedBounds, 'segment bbox');
    }
  });

  it('every segment has a normalised area in [0, 1]', () => {
    for (const seg of segments) {
      expect(seg.area).toBeGreaterThanOrEqual(0);
      expect(seg.area).toBeLessThanOrEqual(1.0);
    }
  });

  it('segments are sorted largest-area-first', () => {
    for (let i = 1; i < segments.length; i++) {
      expect(segments[i - 1].area).toBeGreaterThanOrEqual(segments[i].area);
    }
  });

  it('results are deterministic — same image produces same segment set', async () => {
    const segments2 = await segmentImage(fileA, { segmenterUrl: SEGMENTER_URL });
    expect(segments2.length).toBe(segments.length);

    for (let i = 0; i < segments.length; i++) {
      const a = segments[i];
      const b = segments2[i];
      expect(a.area).toBeCloseTo(b.area, 5);
      expect(a.normalizedBounds).toEqual(b.normalizedBounds);
    }
  });
});
