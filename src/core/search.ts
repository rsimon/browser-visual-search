import { embedImage } from '../embedding/embed.js';
import { nearestNeighbours } from './similarity.js';

import type { BBox, SearchOptions, SearchResult } from '../types.js';
import type { VisualSearchIndex } from '../indexing/store.js';

/**
 * Embed a query image/region and return the top-K nearest segments from the index.
 * @param file The query image file
 * @param bbox Optional bounding box for region
 * @param index The loaded visual search index
 * @param options Search options including model URLs and topK
 * @returns Array of search results
 */
export async function search(
  file: File,
  bbox: BBox | undefined,
  index: VisualSearchIndex,
  options: SearchOptions & { embedderUrl: string; executionProviders?: string[] } = { embedderUrl: '' }
): Promise<SearchResult[]> {
  const queryEmbedding = await embedImage(file, bbox, options);
  return nearestNeighbours(queryEmbedding, index, options.topK ?? 20);
}