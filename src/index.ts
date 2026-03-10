/**
 * browser-visual-search
 * Public package entry point.
 */

export { loadVisualSearch, deserializeIndex } from './visual-search.js';

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