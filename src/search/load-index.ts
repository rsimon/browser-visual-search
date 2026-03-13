import { embedImage } from '../embedding/embed.js';
import type { BBox, IndexedImage, SearchOptions, SearchResult, VisualSearchIndex } from '../types.js';

const DIR_NAME   = '.visual-search';
const INDEX_FILE = 'index.json';
const EMBED_FILE = 'embeddings.bin';

export const EMBEDDING_DIM = 512;

export interface LoadIndexOptions {

  embedderUrl: string;

  executionProviders?: string[];

}

interface VisualSearchIndexData {
  
  model: string;

  updated: string;

  images: IndexedImage[];

}

const nearestNeighbours = (
  query: Float32Array,
  embeddings: Float32Array,
  images: IndexedImage[],
  topK = 20,
): SearchResult[] => {
  const results: SearchResult[] = [];

  for (const img of images) {
    for (const seg of img.segments) {
      const offset = seg.embeddingRow * EMBEDDING_DIM;
      let score = 0;
      for (let k = 0; k < EMBEDDING_DIM; k++)
        score += query[k] * embeddings[offset + k];
      results.push({ imageId: img.imageId, bbox: seg.bbox, area: seg.area, score });
    }
  }

  return results
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export const createIndex = (
  images: IndexedImage[],
  embeddings: Float32Array,
  dirHandle: FileSystemDirectoryHandle,
  opts: LoadIndexOptions,
): VisualSearchIndex => {
  const imageMap = new Map(images.map(img => [img.imageId, img]));

  return {
    images,
    embeddings,
    dirHandle,

    getImage(imageId: string): IndexedImage | undefined {
      return imageMap.get(imageId);
    },

    async query(file: File, bbox?: BBox, options?: SearchOptions): Promise<SearchResult[]> {
      const queryEmbedding = await embedImage(file, bbox, {
        embedderUrl: opts.embedderUrl,
        executionProviders: opts.executionProviders,
      });

      return nearestNeighbours(queryEmbedding, embeddings, images, options?.topK);
    }
  }
}

/**
 * Load an existing visual search index from a directory handle.
 * @param dirHandle The directory handle containing index.json and embeddings.bin
 * @returns The loaded index
 */
export const loadIndex = async (dirHandle: FileSystemDirectoryHandle, opts: LoadIndexOptions): Promise<VisualSearchIndex> => {
  const vsDir = await dirHandle.getDirectoryHandle(DIR_NAME);

  const jsonHandle = await vsDir.getFileHandle(INDEX_FILE);
  const jsonFile   = await jsonHandle.getFile();
  const jsonText   = await jsonFile.text();
  const { images }  = JSON.parse(jsonText) as VisualSearchIndexData;

  const binHandle  = await vsDir.getFileHandle(EMBED_FILE);
  const binFile    = await binHandle.getFile();
  const binBuffer  = await binFile.arrayBuffer();
  const embeddings = new Float32Array(binBuffer);

  // Validate length
  const totalSegments = images.reduce((n, img) => n + img.segments.length, 0);
  if (embeddings.length !== totalSegments * EMBEDDING_DIM)
    throw new Error(`embeddings.bin length mismatch: expected ${totalSegments * EMBEDDING_DIM}, got ${embeddings.length}`);

  return createIndex(images, embeddings, dirHandle, opts);
}