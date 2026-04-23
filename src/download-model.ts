import { ModelLoadStatus } from './types.js';

/**
 * Adapted from:
 * https://github.com/annotorious/plugin-segment-anything/blob/main/src/sam2/utils
 */

const getFilename = (url: string): string => {
  const cleanUrl = url.split(/[?#]/)[0];
  return cleanUrl.substring(cleanUrl.lastIndexOf('/') + 1);
}

const fetchWithProgress = async (
  url: string, 
  onProgress?: (progress: ModelLoadStatus) => void
): Promise<ArrayBuffer> => {
  const response = await fetch(url);
  
  const contentLength = response.headers.get('Content-Length');
  const total = contentLength ? parseInt(contentLength, 10) : undefined;

  const reader = response.body?.getReader();
  
  if (!reader)
    throw new Error('Could not get reader from response body');

  const chunks: Uint8Array[] = [];

  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    if (value) {
      chunks.push(value);
      loaded += value.length;

      onProgress?.({ status: 'downloading', loaded, total });
    }
  }

  // Combine chunks into a single ArrayBuffer
  const buffer = new ArrayBuffer(loaded);
  const uint8Array = new Uint8Array(buffer);
  
  let position = 0;

  for (const chunk of chunks) {
    uint8Array.set(chunk, position);
    position += chunk.length;
  }

  return buffer;
}

export const isModelCached = async (url: string) => {
  const root = await navigator.storage.getDirectory();
  const filename = getFilename(url);
  const handle = await root
    .getFileHandle(filename)
    .catch(() => { 
      // Do nothing - expected if the file hasn't 
      // yet been downloaded. 
    });

  return Boolean(handle);
}

/** 
 * Load a model file from cache or URL.
 */ 
export const loadModel = async (
  url: string,
  onProgress?: (progress: ModelLoadStatus) => void
): Promise<ArrayBuffer> => {
  const root = await navigator.storage.getDirectory();

  const filename = getFilename(url);

  const handle = await root
    .getFileHandle(filename)
    .catch(() => { 
      // Do nothing - expected if the file hasn't 
      // yet been downloaded. 
    });

  if (handle) {
    const file = await handle.getFile();
    console.log(`[browser-visual-search] Cached: ${filename.substring(0, filename.indexOf('.'))}`);
    onProgress?.({ status: 'cached' });

    if (file.size > 10000000) {
      const buffer = await file.arrayBuffer();
      onProgress?.({ status: 'ready' });
      return buffer;
    } else {
      // Something's off - delete this file
      root.removeEntry(filename);
    }
  }

  try {
    console.log(`[browser-visual-search] Downloading ${filename}`);
    const buffer = await fetchWithProgress(url, onProgress);

    const fileHandle = await root.getFileHandle(filename, { create: true });

    console.log(`[browser-visual-search] Writing to cache`);
    const writable = await fileHandle.createWritable();
    await writable.write(buffer);
    await writable.close();

    onProgress?.({ status: 'ready' });
    return buffer;
  } catch (error) {
    console.error(error);
    throw new Error(`Download failed: ${url}`);
  }
}