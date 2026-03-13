export const SEGMENTER_URL = '/models/fastsam-s.onnx';
export const EMBEDDER_URL  = '/models/clip-vit-b32-visual-int8.onnx';

export const FIXTURE_IMAGES = [
  '/fixtures/image-a.jpg',
  '/fixtures/image-b.jpg',
  '/fixtures/image-c.jpg',
] as const;

// ── Fixture helpers ────────--──────────────────────────────────────────

export const fetchFixture = async (url: string): Promise<File> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch fixture: ${url} (${res.status})`);
  const blob = await res.blob();
  const name = url.split('/').at(-1) ?? 'fixture';
  return new File([blob], name, { type: blob.type });
}

// ── Assertion helpers ──────────────────────────────────────────────────

export const assertInRange = (value: number, min: number, max: number, label = 'value') => {
  if (!isFinite(value) || value < min || value > max)
    throw new Error(`Expected ${label} in [${min}, ${max}], got ${value}`);
}

export const assertValidBBox = (bbox: number[], label = 'bbox') => {
  if (bbox.length !== 4) throw new Error(`${label}: expected 4 elements, got ${bbox.length}`);
  const [x, y, w, h] = bbox;
  assertInRange(x, 0, 1, `${label}[x]`);
  assertInRange(y, 0, 1, `${label}[y]`);
  assertInRange(w, 0, 1, `${label}[w]`);
  assertInRange(h, 0, 1, `${label}[h]`);
  if (x + w > 1.001) throw new Error(`${label}: x+w=${x+w} exceeds 1`);
  if (y + h > 1.001) throw new Error(`${label}: y+h=${y+h} exceeds 1`);
}