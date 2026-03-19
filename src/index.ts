
export { embedImage, embedBatch } from './embedding/index.js';
export { segmentImage } from './segmentation/index.js';
export { buildFromDirectory } from './search-index/build-from-directory.js';
export { indexExists, openIndex } from './search-index/index.js';

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
  BuildFromDirectoryOptions,
  BuildFromDirectoryProgress
} from './search-index/build-from-directory.js';

export type {
  BuildIndexOptions,
  LoadIndexOptions
} from './search-index/index.js';