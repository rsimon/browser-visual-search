import { embedBatch } from '../embedding/embed.js';
import { segmentImage } from '../segmentation/index.js';
import { createIndex, openIndex, type BuildIndexOptions } from './index.js';
import type { IndexedImage, IndexedImageSegment, VisualSearchIndex } from '../types.js';

export interface BuildFromDirectoryOptions extends BuildIndexOptions {

  forceReindex?: boolean;

  onProgress?: (progress: BuildFromDirectoryProgress) => void;

}

export type BuildFromDirectoryProgress =
  | { phase: 'loading' | 'scanning' | 'saving' | 'done' }
  | { phase: 'indexing'; total: number; completed: number; currentFile?: string };

interface ImageInput {

  id: string;

  file: File;

}

const collectImages = async (dirHandle: FileSystemDirectoryHandle, path: string = ''): Promise<ImageInput[]> => {
  const images: ImageInput[] = [];
  const entries = (dirHandle as any).entries();

  for await (const [name, handle] of entries) {
    const fullPath = path ? `${path}/${name}` : name;
    if (handle.kind === 'file') {
      const ext = name.toLowerCase().split('.').pop();
      if (['jpg', 'jpeg', 'png', 'webp'].includes(ext ?? '')) {
        const file = await handle.getFile();
        images.push({ id: fullPath, file });
      }
    } else if (handle.kind === 'directory') {
      images.push(...await collectImages(handle, fullPath));
    }
  }

  return images;
}

export const buildFromDirectory = async (
  dirHandle: FileSystemDirectoryHandle,
  opts: BuildFromDirectoryOptions
): Promise<VisualSearchIndex> => {
  const { onProgress, forceReindex } = opts;

  // 1. Try to load existing index
  onProgress?.({ phase: 'loading' });
  
  let existingImages: IndexedImage[] = [];
  let existingEmbeddings: Float32Array[] = [];

  try {
    const existing = await openIndex(dirHandle, opts);
    existingImages    = [...existing.images];
    existingEmbeddings = [...existing.embeddings];
  } catch {
    // No index yet — start fresh
  }

  // 2. Scan directory
  onProgress?.({ phase: 'scanning' });
  const allImages = await collectImages(dirHandle);
  const allImageIds = new Set(allImages.map(img => img.id));

  // 3. Prune images no longer on disk
  const keptImages = existingImages.filter(img => allImageIds.has(img.imageId));

  // 4. Determine which images need indexing
  const indexedIds = new Set(keptImages.map(img => img.imageId));
  const toIndex = forceReindex
    ? allImages
    : allImages.filter(img => !indexedIds.has(img.id));

  // 5. Index new images
  const newImages: IndexedImage[] = [];
  const newEmbeddings: Float32Array[] = [];

  for (let i = 0; i < toIndex.length; i++) {
    const input = toIndex[i];
    onProgress?.({ phase: 'indexing', total: toIndex.length, completed: i, currentFile: input.id });

    const segOpts = { 
      segmenterUrl: opts.segmenterUrl, 
      executionProviders: opts.executionProviders,
      maxDetections: opts.maxDetectionsPerImage
    };

    const detections = await segmentImage(input.file, segOpts);
    const bitmap     = await createImageBitmap(input.file);
    const embeddings: Float32Array[] = await embedBatch(bitmap, detections.map(d => d.bbox), opts);
    bitmap.close();

    const segments: IndexedImageSegment[] = detections.map((det, j) => ({
      bbox:         det.bbox,
      area:         det.area,
      embeddingRow: j
    }));

    console.debug(`[browser-local-search] ${input.id}: ${embeddings.length} embedding vectors`);

    newImages.push({ imageId: input.id, indexedAt: new Date().toISOString(), segments });
    newEmbeddings.push(...embeddings);
  }

  // 6. Merge kept + new
  // Re-map embeddingRows for kept images, then append new
  const mergedImages: IndexedImage[] = [];
  const mergedEmbeddingVecs: Float32Array[] = [];
  let row = 0;

  const baseImages = forceReindex ? [] : keptImages;

  for (const img of baseImages) {
    const remappedSegments = img.segments.map(seg => {
      mergedEmbeddingVecs.push(existingEmbeddings[seg.embeddingRow]);
      return { ...seg, embeddingRow: row++ };
    });

    mergedImages.push({ ...img, segments: remappedSegments });
  }

  let newEmbeddingsOffset = 0;

  for (const img of newImages) {
    const remappedSegments = img.segments.map(seg => {
      mergedEmbeddingVecs.push(newEmbeddings[newEmbeddingsOffset + seg.embeddingRow]);
      return {
        ...seg,
        embeddingRow: row++
      };
    });

    mergedImages.push({ ...img, segments: remappedSegments });
    newEmbeddingsOffset += img.segments.length;
  }

  // 7. Serialize
  onProgress?.({ phase: 'saving' });

  const index = createIndex(mergedImages, mergedEmbeddingVecs, dirHandle, opts);
  await index.save();

  onProgress?.({ phase: 'done' });
  return index;
}

