// allow importing the helper package which has no types
declare module 'compute-cosine-similarity' {
  function cosineSimilarity(a: Float32Array, b: Float32Array): number;
  export default cosineSimilarity;
}
