/**
 * Brute-force cosine nearest-neighbour search.
 *
 * All embeddings are unit-normalised (enforced at embed time), so
 * cosine similarity reduces to a dot product — no division needed.
 *
 * This runs entirely in JS with no ONNX involvement.
 */

import type { SearchResult } from './types.js';
import type { VisualSearchIndex } from './index-store.js';

const EMBEDDING_DIM = 512;

/**
 * Dot product of two unit-normalised vectors — equivalent to cosine similarity.
 */
function dotProduct(a: Float32Array, b: Float32Array, bOffset: number): number {
  let sum = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    sum += a[i] * b[bOffset + i];
  }
  return sum;
}

/**
 * Search the index for segments most similar to queryEmbedding.
 *
 * @param queryEmbedding  512-dim unit-normalised Float32Array
 * @param index           VisualSearchIndex (in-memory)
 * @param topK            Number of results to return (default 20)
 */
export function nearestNeighbours(
  queryEmbedding: Float32Array,
  index: VisualSearchIndex,
  topK = 20,
): SearchResult[] {
  const { embeddings, images } = index;
  const results: SearchResult[] = [];

  for (const image of images) {
    for (const seg of image.segments) {
      const score = dotProduct(queryEmbedding, embeddings, seg.embeddingRow * EMBEDDING_DIM);
      results.push({
        imageId: image.imageId,
        bbox:    seg.bbox,
        area:    seg.area,
        score,
      });
    }
  }

  // Sort descending by score, return top K
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}