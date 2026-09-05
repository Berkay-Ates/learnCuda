# 202 — Memory Model & Matrices

Goal: matrices are the standard vehicle for learning the CUDA memory
hierarchy, because the naive version is embarrassingly parallel but
memory-bound, and the optimized version forces you to use shared memory.

## Projects (do in order)

1. **matrix_add** — warm-up. 2D thread indexing: map `(blockIdx, threadIdx)`
   in x and y to a `(row, col)` in the matrix. Use a 2D grid/block
   (`dim3`) instead of 1D.

2. **matrix_mul_naive** — one thread computes one output element `C[row][col]`
   by looping over the shared dimension `k` and reading directly from global
   memory each time. It works, but every thread re-reads the same rows/columns
   from slow global memory — profile it and note the runtime.

3. **matrix_mul_tiled** — same result, but each thread block first cooperatively
   loads a `TILE x TILE` tile of A and B into `__shared__` memory, calls
   `__syncthreads()`, then all threads in the block compute their partial sums
   from the fast on-chip copy before moving to the next tile. Compare runtime
   against the naive version — this is usually where CUDA "clicks."

## What to reuse from 101

- `CUDA_CHECK` macro from [common/cuda_check.cuh](../common/cuda_check.cuh)
- The alloc → H2D → launch → D2H → verify → free skeleton from `vector_add`

## Concepts you should be able to explain by the end

- Why global memory access dominates the naive kernel's time (memory-bound
  vs. compute-bound)
- What `__syncthreads()` is protecting against if you remove it from the
  tiled kernel (try it — you'll get wrong answers, non-deterministically)
- Why tile size is usually chosen as a multiple of 32 and constrained by
  `sharedMemPerBlock` from your `device_query` output
