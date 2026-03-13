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

  readonly images: ReadonlyArray<IndexedImage>;

  readonly embeddings: Float32Array;

  readonly dirHandle: FileSystemDirectoryHandle;

  query(file: File, bbox?: BBox, options?: SearchOptions): Promise<SearchResult[]>; 

  getImage(imageId: string): IndexedImage | undefined;

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

