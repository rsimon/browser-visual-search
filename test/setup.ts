/**
 * tests/setup.ts
 * --------------
 * Loads the VisualSearch instance once and exposes shared helpers.
 *
 * Assumes models are served from /models/ (placed in assets/models/).
 * Assumes test fixture images are served from /fixtures/ (assets/fixtures/).
 */

import * as ort from 'onnxruntime-web';

// ORT Web needs its WASM files served as static assets.
// Vite serves node_modules via /@fs/ in dev mode.
ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';

import { loadVisualSearch } from '../src/visual-search.js';
import type { VisualSearch } from '../src/visual-search.js';

// ── Model URLs (served by Vitest's static server from public/) ────────────────

export const SEGMENTER_URL = '/models/fastsam-s.onnx';
export const EMBEDDER_URL  = '/models/clip-vit-b32-visual.onnx';

// ── Fixture image URLs ────────────────────────────────────────────────────────
// Tests reference them by name; helpers below fetch them as Files.

export const FIXTURE_IMAGES = [
  '/fixtures/image-a.jpg',
  '/fixtures/image-b.png',
  '/fixtures/image-c.jpg',
] as const;

// ── Singleton VS instance ─────────────────────────────────────────────────────

let _vs: VisualSearch | null = null;

export async function getVS(): Promise<VisualSearch> {
  if (!_vs) {
    _vs = await loadVisualSearch({
      segmenterUrl: SEGMENTER_URL,
      embedderUrl:  EMBEDDER_URL,
      // Use WebGL if available, fall back to WASM
      executionProviders: ['webgl', 'wasm'],
    });
  }
  return _vs;
}

export async function fetchFixture(url: string): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch fixture: ${url} (${res.status})`);
  const blob = await res.blob();
  const name = url.split('/').at(-1) ?? 'fixture';
  return new File([blob], name, { type: blob.type });
}

// ── Simple assertion helpers ──────────────────────────────────────────────────

/** Assert a value is a finite number in [min, max]. */
export function assertInRange(value: number, min: number, max: number, label = 'value'): void {
  if (!isFinite(value) || value < min || value > max) {
    throw new Error(`Expected ${label} in [${min}, ${max}], got ${value}`);
  }
}

/** Assert a BBox is a valid normalised [x, y, w, h]. */
export function assertValidBBox(bbox: number[], label = 'bbox'): void {
  if (bbox.length !== 4) throw new Error(`${label}: expected 4 elements, got ${bbox.length}`);
  const [x, y, w, h] = bbox;
  assertInRange(x, 0, 1, `${label}[x]`);
  assertInRange(y, 0, 1, `${label}[y]`);
  assertInRange(w, 0, 1, `${label}[w]`);
  assertInRange(h, 0, 1, `${label}[h]`);
  if (x + w > 1.001) throw new Error(`${label}: x+w=${x+w} exceeds 1`);
  if (y + h > 1.001) throw new Error(`${label}: y+h=${y+h} exceeds 1`);
}