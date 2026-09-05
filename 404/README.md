# 404 — Concurrency & Advanced Tricks

Goal: move beyond a single kernel launch — overlap work, manage memory more
flexibly, and handle 2D neighborhoods.

## Projects (do in order)

1. **streams_overlap** — split a large vector-add-style workload into
   chunks, and use multiple `cudaStream_t` streams so `cudaMemcpyAsync`
   (H2D for chunk N+1) overlaps with the kernel execution (for chunk N).
   Time it against a single-stream version to see the overlap win — this
   needs `asyncEngineCount >= 1` from your device_query output (it is, on
   the 3060).

2. **unified_memory** — take the vector_add or matrix_mul from 101/202 and
   rewrite it using `cudaMallocManaged` instead of explicit
   `cudaMalloc`/`cudaMemcpy`. Compare code complexity and performance
   against the explicit version — unified memory is simpler but the driver's
   page migration isn't free.

3. **convolution_2d** — box blur (or Sobel edge detection) on a 2D image
   buffer. Each output pixel depends on a small neighborhood, so the
   shared-memory tile needs a "halo" border loaded from neighboring blocks —
   this is the natural extension of 202's tiled matmul to a 2D stencil
   pattern.

## Concepts you should be able to explain by the end

- Why streams need `cudaMemcpyAsync` + pinned host memory
  (`cudaHostAlloc`/`cudaMallocHost`) to actually overlap, not just any
  memcpy
- The halo/ghost-cell pattern for stencil kernels
- When unified memory helps (development speed, irregular access) vs. hurts
  (page fault overhead on tight loops)
