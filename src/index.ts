export { loadIndex } from './indexing/load.js';
export { search } from './core/search.js';
export { embedImage, embedBatch } from './embedding/embed.js';
export { buildIndex } from './indexing/builder.js';
export { segmentImage } from './segmentation/segment.js';

export type {
  BBox,
  ImageInput,
  Segment,
  SearchResult,
  BuildIndexOptions,
  SearchOptions,
  IndexImageOptions,
} from './types.js';

export type { VisualSearchIndex } from './indexing/store.js';