/**
 * Normalised bounding box: [x, y, w, h] in [0, 1], origin top-left.
 */
export type BBox = [number, number, number, number];

/**
 * Input to indexImage() and buildIndex().
 * The caller is responsible for supplying a stable, unique ID.
 */
export interface ImageInput {
  id: string;
  file: File;
}

/**
 * A single segment returned by indexImage().
 * The imageId is not included — the caller supplied it and owns it.
 */
export interface Segment {
  bbox: BBox;
  area: number;
  embedding: Float32Array;
}

/**
 * A segment as stored in the index (embedding referenced by row, not inline).
 */
export interface PersistedSegment {
  bbox: BBox;
  area: number;
  embeddingRow: number;
}

/**
 * A single indexed image as stored in index.json.
 */
export interface PersistedImage {
  imageId: string;
  indexedAt: string;
  segments: PersistedSegment[];
}

/**
 * The full index.json structure.
 */
export interface PersistedIndex {
  version: 1;
  model: string;
  updatedAt: string;
  images: PersistedImage[];
}

/**
 * A single search result.
 */
export interface SearchResult {
  imageId: string;
  bbox: BBox;
  area: number;
  score: number; // cosine similarity in [0, 1]
}

/**
 * Options passed to loadVisualSearch().
 */
export interface LoadOptions {
  segmenterUrl: string;
  embedderUrl: string;
  /** Defaults to ['wasm']. Pass ['webgpu', 'wasm'] to prefer WebGPU. */
  executionProviders?: string[];
}

/**
 * Options for buildIndex().
 */
export interface BuildIndexOptions {
  onProgress?: (completed: number, total: number) => void;
}

/**
 * Options for search().
 */
export interface SearchOptions {
  topK?: number; // default 20
}