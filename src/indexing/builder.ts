import { segmentImage } from '../segmentation/segment.js';
import { embedBatch } from '../embedding/embed.js';
import { createIndexBuilder } from './store.js';

import type { ImageInput, BuildIndexOptions, Segment } from '../types.js';
import type { VisualSearchIndex } from './store.js';

/**
 * Recursively collect all image files from a directory handle.
 * @param dirHandle The directory handle
 * @param path Current path for ID generation
 * @returns Array of ImageInput
 */
async function collectImages(dirHandle: FileSystemDirectoryHandle, path = ''): Promise<ImageInput[]> {
  const images: ImageInput[] = [];

  // TypeScript types are incomplete for File System Access API
  const entries = (dirHandle as any).entries();
  for await (const [name, handle] of entries) {
    const fullPath = path ? `${path}/${name}` : name;

    if (handle.kind === 'file') {
      // Check if it's an image file
      const ext = name.toLowerCase().split('.').pop();
      if (['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext || '')) {
        const file = await handle.getFile();
        images.push({ id: fullPath, file });
      }
    } else if (handle.kind === 'directory') {
      // Recurse into subdirectory
      const subImages = await collectImages(handle, fullPath);
      images.push(...subImages);
    }
  }

  return images;
}

/**
 * Build a visual search index from a directory of images.
 * Recursively walks subfolders to find all images.
 * @param dirHandle The directory handle containing images
 * @param options Build options including model URLs and progress callbacks
 * @returns The built index
 */
export async function buildIndex(
  dirHandle: FileSystemDirectoryHandle,
  options: BuildIndexOptions & { segmenterUrl: string; embedderUrl: string; executionProviders?: string[] } = {
    segmenterUrl: '',
    embedderUrl: ''
  }
): Promise<VisualSearchIndex> {
  if (!options.segmenterUrl || !options.embedderUrl) {
    throw new Error('segmenterUrl and embedderUrl are required');
  }

  const images = await collectImages(dirHandle);
  const builder = createIndexBuilder();
  const { onProgress, imageProgress } = options;

  for (let i = 0; i < images.length; i++) {
    const input = images[i];

    // Segment the image
    const detections = await segmentImage(input.file, options);
    imageProgress?.onSegmentationDone?.(detections.length);

    // Load bitmap for embedding
    const bitmap = await createImageBitmap(input.file);

    // Embed all segments
    const embeddings = await embedBatch(bitmap, detections.map(d => d.bbox), options);
    imageProgress?.onEmbeddingDone?.(detections.length);

    // Create segments
    const segments: Segment[] = detections.map((det, j) => ({
      bbox: det.bbox,
      area: det.area,
      embedding: embeddings[j],
    }));

    builder.addImage(input.id, segments);
    bitmap.close();

    onProgress?.(i + 1, images.length);
  }

  return builder.build();
}