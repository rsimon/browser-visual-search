export type BBox = [number, number, number, number];

export interface Segment {

  bbox: BBox;

  area: number;

  embedding: Float32Array;

}

export interface IndexedImageSegment {

  bbox: BBox;

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

}

export interface SearchOptions {

  topK?: number;
  
}

export interface SearchResult {

  imageId: string;

  bbox: BBox;

  area: number;

  score: number;

}

