/**
 * browser-visual-search
 * ─────────────────────
 * Convenience wrapper providing a simple `loadVisualSearch()` entry point.
 *
 * Internally it builds a segmenter and embedder using the low-level
 * factories in `segment.ts` and `embed.ts`.  The object returned here has the
 * same shape as the previous library, ensuring existing code continues to
 * function.  Developers are encouraged to bypass this helper and construct the
 * factories directly when they need finer control.
 */

import { createIndexBuilder, deserializeIndex } from './index-store.js';
import { nearestNeighbours } from './search.js';
import { createSegmenter } from './segment.js';
import { createEmbedder } from './embed.js';

import type {
  BBox,
  BuildIndexOptions,
  IndexImageOptions,
  ImageInput,
  LoadOptions,
  SearchOptions,
  SearchResult,
  Segment,
} from './types.js';
import type { VisualSearchIndex } from './index-store.js';

// re-export types for convenience
export type {
  BBox,
  BuildIndexOptions,
  IndexImageOptions,
  ImageInput,
  LoadOptions,
  SearchOptions,
  SearchResult,
  Segment,
  VisualSearchIndex,
};
export { deserializeIndex };

export interface VisualSearch {
  indexImage(input: ImageInput, progress?: IndexImageOptions): Promise<Segment[]>;
  embedRegion(file: File, bbox?: BBox): Promise<Float32Array>;
  buildIndex(inputs: ImageInput[], options?: BuildIndexOptions): Promise<VisualSearchIndex>;
  search(
    file: File,
    bbox: BBox | undefined,
    index: VisualSearchIndex,
    options?: SearchOptions,
  ): Promise<SearchResult[]>;
}

export async function loadVisualSearch(options: LoadOptions): Promise<VisualSearch> {
  const segmenter = createSegmenter({
    modelUrl: options.segmenterUrl,
    executionProviders: options.executionProviders,
  });
  const embedder = createEmbedder({
    modelUrl: options.embedderUrl,
    executionProviders: options.executionProviders,
  });

  async function indexImage(input: ImageInput, progress?: IndexImageOptions): Promise<Segment[]> {
    const detections = await segmenter.segment(input.file);
    progress?.onSegmentationDone?.(detections.length);

    const embeddings = await embedder.embedBatch(input.file, detections.map(d => d.bbox));
    progress?.onEmbeddingDone?.(detections.length);

    return detections.map((det, i) => ({
      bbox: det.bbox,
      area: det.area,
      embedding: embeddings[i],
    }));
  }

  async function embedRegion(file: File, bbox?: BBox): Promise<Float32Array> {
    return embedder.embed(file, bbox);
  }

  async function buildIndex(
    inputs: ImageInput[],
    options: BuildIndexOptions = {},
  ): Promise<VisualSearchIndex> {
    const builder = createIndexBuilder();
    const { onProgress, imageProgress } = options;

    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const segments = await indexImage(input, imageProgress);
      builder.addImage(input.id, segments);
      onProgress?.(i + 1, inputs.length);
    }

    return builder.build();
  }

  async function search(
    file: File,
    bbox: BBox | undefined,
    index: VisualSearchIndex,
    options: SearchOptions = {},
  ): Promise<SearchResult[]> {
    const queryEmbedding = await embedder.embed(file, bbox);
    return nearestNeighbours(queryEmbedding, index, options.topK ?? 20);
  }

  return { indexImage, embedRegion, buildIndex, search };
}
