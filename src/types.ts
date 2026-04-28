export type ModelLoadStatus =
  | { status: 'cached' }
  | { status: 'downloading'; loaded: number; total?: number } 
  | { status: 'download_complete' }
  | { status: 'model_ready'};

export type BBox = [number, number, number, number];

export interface IndexedImageSegment {

  normalizedBounds: BBox;

  pxBounds: BBox;

  area: number;

  embeddingRow: number;

}

export interface IndexedImage {

  imageId: string;

  indexedAt: string;

  segments: IndexedImageSegment[];

}

export interface VisualSearchIndex {

  readonly dirHandle: FileSystemDirectoryHandle;

  readonly images: ReadonlyArray<IndexedImage>;

  readonly embeddings: Float32Array[];

  addToIndex(image: File, id: string): Promise<void>;

  getImage(imageId: string): IndexedImage | undefined;

  query(image: Blob, bbox?: BBox, options?: SearchOptions): Promise<SearchResult[]>; 

  save(): Promise<void>;

  downloadSegmentationModel(onProgress: (progress: ModelLoadStatus) => void): Promise<void>;
  
  downloadEmbeddingModel(onProgress: (progress: ModelLoadStatus) => void): Promise<void>;

}

export interface SearchOptions {

  topK?: number;
  
}

export interface SearchResult {

  imageId: string;

  normalizedBounds: BBox;

  pxBounds: BBox;

  area: number;

  score: number;

}

