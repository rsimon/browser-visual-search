
export { embedImage, embedBatch } from './embedding/index.js';
export { segmentImage } from './segmentation/index.js';
export { buildIndex } from './search/build-index.js';
export { loadIndex } from './search/load-index.js';
export { search } from './search/query.js';

export type {
  BBox,
  IndexedImage,
  IndexedImageSegment,
  SearchOptions,
  SearchResult,
  Segment,
  VisualSearchIndex
} from './types.js';

export type {
  BuildOptions,
  BuildProgress
} from './search/build-index.js';

export type {
  LoadIndexOptions
} from './search/load-index.js';