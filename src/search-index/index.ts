import { embedBatch, embedImage } from '../embedding/embed.js';
import { segmentImage } from '../segmentation/segment.js';
import type { 
  BBox, 
  IndexedImage, 
  IndexedImageSegment, 
  SearchOptions, 
  SearchResult, 
  VisualSearchIndex 
} from '../types.js';

const DIR_NAME   = ['.visual-search', 'visual-search'];
const INDEX_FILE = 'index.json';
const EMBED_FILE = 'embeddings.bin';

export const EMBEDDING_DIM = 512;

export interface LoadIndexOptions {

  embedderUrl: string;

  executionProviders?: string[];

  create?: boolean;

}

export interface BuildIndexOptions extends LoadIndexOptions {

  segmenterUrl: string;

}

interface VisualSearchIndexData {
  
  model: string;

  updated: string;

  images: IndexedImage[];

}

const getDirectoryHandle = async (
  dirHandle: FileSystemDirectoryHandle,
  options?: FileSystemGetDirectoryOptions,
): Promise<FileSystemDirectoryHandle> => {
  let lastError: unknown;

  for (const name of DIR_NAME) {
    try {
      return await dirHandle.getDirectoryHandle(name, options);
    } catch (err) {
      lastError = err;
    }
  }

  throw lastError;
};

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

const appendEmbeddings = (
  current: Float32Array,
  toAdd: Float32Array[],
): Float32Array => {
  const totalFloats = current.length + toAdd.reduce((n, c) => n + c.length, 0);

  const next = new Float32Array(totalFloats);
  next.set(current, 0);

  let offset = current.length;

  for (const chunk of toAdd) {
    next.set(chunk, offset);
    offset += chunk.length;
  }

  return next;
}

export const indexExists = async (
  dirHandle: FileSystemDirectoryHandle
): Promise<boolean> => {
  try {
    const vsDir = await getDirectoryHandle(dirHandle);
    await vsDir.getFileHandle(INDEX_FILE);
    await vsDir.getFileHandle(EMBED_FILE);
    return true;
  } catch (err) {
    if (err instanceof DOMException && err.name === 'NotFoundError') {
      return false;
    }
    throw err; // re-throw unexpected errors (e.g. SecurityError, TypeError)
  }
}

export const createIndex = (
  images: IndexedImage[],
  embeddings: Float32Array,
  dirHandle: FileSystemDirectoryHandle,
  opts: LoadIndexOptions | BuildIndexOptions
): VisualSearchIndex => {
  const imageMap = new Map(images.map(img => [img.imageId, img]));

  let _embeddings = embeddings;

  return {
    dirHandle,

    images,
    
    get embeddings() { return _embeddings; },

    async addToIndex(image: File, id: string) {
      if (!('segmenterUrl' in opts)) throw new Error('addToIndex requires a segmenter model');

      const detections = await segmentImage(image, opts);
      const bitmap     = await createImageBitmap(image);
      const vecs = await embedBatch(bitmap, detections.map(d => d.bbox), opts);
      bitmap.close();

      const nextRow = _embeddings.length / EMBEDDING_DIM;

      const segments: IndexedImageSegment[] = detections.map((det, j) => ({
        bbox: det.bbox,
        area: det.area,
        embeddingRow: nextRow + j
      }));

      images.push({
        imageId: id,
        indexedAt: new Date().toISOString(),
        segments
      });

      _embeddings = appendEmbeddings(_embeddings, vecs);
    },

    getImage(imageId: string): IndexedImage | undefined {
      return imageMap.get(imageId);
    },

    async query(file: File, bbox?: BBox, options?: SearchOptions): Promise<SearchResult[]> {
      const queryEmbedding = await embedImage(file, bbox, {
        embedderUrl: opts.embedderUrl,
        executionProviders: opts.executionProviders,
      });

      return nearestNeighbours(queryEmbedding, _embeddings, images, options?.topK);
    },

    async save() {
      const vsDir = await getDirectoryHandle(dirHandle, { create: true });

      const jsonHandle = await vsDir.getFileHandle('index.json', { create: true });
      const jsonWriter = await jsonHandle.createWritable();
      await jsonWriter.write(JSON.stringify({ version: 1, updatedAt: new Date().toISOString(), images }, null, 2));
      await jsonWriter.close();

      const binHandle = await vsDir.getFileHandle('embeddings.bin', { create: true });
      const binWriter = await binHandle.createWritable();
      await binWriter.write(_embeddings.buffer as ArrayBuffer);
      await binWriter.close();
    }
  }
}

/**
 * Load an existing visual search index from a directory handle.
 * @param dirHandle The directory handle containing index.json and embeddings.bin
 * @returns The loaded index
 */
export const openIndex = async (
  dirHandle: FileSystemDirectoryHandle, 
  opts: LoadIndexOptions | BuildIndexOptions
): Promise<VisualSearchIndex> => {
  if (!await indexExists(dirHandle)) {
    if (opts.create) return createIndex([], new Float32Array(0), dirHandle, opts);
    throw new Error(`No index found in directory`);
  }
  
  const vsDir = await getDirectoryHandle(dirHandle);

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