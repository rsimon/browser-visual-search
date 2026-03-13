
export { embedImage, embedBatch } from './embedding/index.js';
export { segmentImage } from './segmentation/index.js';
export { buildIndex, loadIndex } from './search/index.js';

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