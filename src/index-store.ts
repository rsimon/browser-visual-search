/**
 * VisualSearchIndex — in-memory index with serialisation to
 * FileSystem API (.visual-search/index.json + embeddings.bin).
 */

import type {
  BBox,
  PersistedImage,
  PersistedIndex,
  PersistedSegment,
  Segment,
} from './types.js';

const DIR_NAME   = '.visual-search';
const INDEX_FILE = 'index.json';
const EMBED_FILE = 'embeddings.bin';

export const MODEL_LABEL = 'clip-vit-b32 / fastsam-s';
const EMBEDDING_DIM = 512;

// ── Internal representation ───────────────────────────────────────────────────

interface InternalImage {
  imageId: string;
  indexedAt: string;
  segments: Array<{
    bbox: BBox;
    area: number;
    embeddingRow: number;
  }>;
}

// ── Index class ───────────────────────────────────────────────────────────────

export interface VisualSearchIndex {
  /** All indexed images. */
  readonly images: ReadonlyArray<{
    imageId: string;
    indexedAt: string;
    segments: ReadonlyArray<{ bbox: BBox; area: number; embeddingRow: number }>;
  }>;

  /**
   * The full embeddings matrix.
   * Row i is a 512-dim unit-normalised Float32Array.
   * Total length = numSegments * 512.
   */
  readonly embeddings: Float32Array;

  /**
   * Write .visual-search/index.json and .visual-search/embeddings.bin
   * into the given directory handle.
   */
  serialize(dirHandle: FileSystemDirectoryHandle): Promise<void>;
}

export function createIndex(
  images: InternalImage[],
  embeddings: Float32Array,
): VisualSearchIndex {
  return {
    images,
    embeddings,

    async serialize(dirHandle: FileSystemDirectoryHandle): Promise<void> {
      // Create (or open) the .visual-search sub-directory
      const vsDir = await dirHandle.getDirectoryHandle(DIR_NAME, { create: true });

      // ── index.json ────────────────────────────────────────────────────────
      const persistedImages: PersistedImage[] = images.map((img) => ({
        imageId:   img.imageId,
        indexedAt: img.indexedAt,
        segments:  img.segments.map((s): PersistedSegment => ({
          bbox:         s.bbox,
          area:         s.area,
          embeddingRow: s.embeddingRow,
        })),
      }));

      const indexPayload: PersistedIndex = {
        version:   1,
        model:     MODEL_LABEL,
        updatedAt: new Date().toISOString(),
        images:    persistedImages,
      };

      const jsonHandle = await vsDir.getFileHandle(INDEX_FILE, { create: true });
      const jsonWriter = await jsonHandle.createWritable();
      await jsonWriter.write(JSON.stringify(indexPayload, null, 2));
      await jsonWriter.close();

      // ── embeddings.bin ────────────────────────────────────────────────────
      const binHandle = await vsDir.getFileHandle(EMBED_FILE, { create: true });
      const binWriter = await binHandle.createWritable();
      await binWriter.write(embeddings.buffer as ArrayBuffer);
      await binWriter.close();
    },
  };
}

// ── Deserialise ───────────────────────────────────────────────────────────────

/**
 * Load a previously serialised index from a directory handle.
 * Looks for .visual-search/index.json and .visual-search/embeddings.bin.
 */
export async function deserializeIndex(
  dirHandle: FileSystemDirectoryHandle,
): Promise<VisualSearchIndex> {
  const vsDir = await dirHandle.getDirectoryHandle(DIR_NAME);

  // ── index.json ────────────────────────────────────────────────────────────
  const jsonHandle = await vsDir.getFileHandle(INDEX_FILE);
  const jsonFile   = await jsonHandle.getFile();
  const jsonText   = await jsonFile.text();
  const persisted  = JSON.parse(jsonText) as PersistedIndex;

  if (persisted.version !== 1) {
    throw new Error(`Unsupported index version: ${persisted.version}`);
  }

  const images: InternalImage[] = persisted.images.map((img) => ({
    imageId:   img.imageId,
    indexedAt: img.indexedAt,
    segments:  img.segments.map((s) => ({
      bbox:         s.bbox as BBox,
      area:         s.area,
      embeddingRow: s.embeddingRow,
    })),
  }));

  // ── embeddings.bin ────────────────────────────────────────────────────────
  const binHandle  = await vsDir.getFileHandle(EMBED_FILE);
  const binFile    = await binHandle.getFile();
  const binBuffer  = await binFile.arrayBuffer();
  const embeddings = new Float32Array(binBuffer);

  // Validate length
  const totalSegments = images.reduce((n, img) => n + img.segments.length, 0);
  if (embeddings.length !== totalSegments * EMBEDDING_DIM) {
    throw new Error(
      `embeddings.bin length mismatch: expected ${totalSegments * EMBEDDING_DIM}, got ${embeddings.length}`,
    );
  }

  return createIndex(images, embeddings);
}

// ── Builder helper (used by the main API) ────────────────────────────────────

export interface IndexBuilder {
  addImage(imageId: string, segments: Segment[]): void;
  build(): VisualSearchIndex;
}

export function createIndexBuilder(): IndexBuilder {
  const images: InternalImage[]     = [];
  const allEmbeddings: Float32Array[] = [];
  let nextRow = 0;

  return {
    addImage(imageId: string, segments: Segment[]): void {
      const indexedAt = new Date().toISOString();
      const persistedSegments = segments.map((s) => {
        allEmbeddings.push(s.embedding);
        return {
          bbox:         s.bbox,
          area:         s.area,
          embeddingRow: nextRow++,
        };
      });
      images.push({ imageId, indexedAt, segments: persistedSegments });
    },

    build(): VisualSearchIndex {
      // Concatenate all embeddings into a single flat Float32Array
      const total = allEmbeddings.length * EMBEDDING_DIM;
      const embeddings = new Float32Array(total);
      allEmbeddings.forEach((vec, i) => {
        embeddings.set(vec, i * EMBEDDING_DIM);
      });
      return createIndex(images, embeddings);
    },
  };
}