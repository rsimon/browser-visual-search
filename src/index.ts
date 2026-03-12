/**
 * browser-visual-search
 * Public package entry point.
 */

export { loadVisualSearch, deserializeIndex } from './visual-search.js';
export { createSegmenter } from './segment.js';
export { createEmbedder } from './embed.js';
export { nearestNeighbours } from './search.js';
export { createIndexBuilder } from './index-store.js';

export type {
  VisualSearch,
  VisualSearchIndex,
  BBox,
  ImageInput,
  Segment,
  SearchResult,
  LoadOptions,
  BuildIndexOptions,
  SearchOptions,
} from './visual-search.js';