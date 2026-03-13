import { deserializeIndex } from './store.js';

import type { VisualSearchIndex } from './store.js';

/**
 * Load an existing visual search index from a directory handle.
 * @param dirHandle The directory handle containing index.json and embeddings.bin
 * @returns The loaded index
 */
export async function loadIndex(dirHandle: FileSystemDirectoryHandle): Promise<VisualSearchIndex> {
  return deserializeIndex(dirHandle);
}