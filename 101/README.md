# 101 — Fundamentals

Goal: get comfortable with the CUDA programming model before optimizing
anything.

## Projects (do in order)

1. **[00_device_query](00_device_query/device_query.cu)** — print your GPU's
   actual specs. Write down the SM count and max threads per block; you'll
   use these numbers later.
2. **[01_hello_world](01_hello_world/hello.cu)** — launch a kernel, print
   `threadIdx`/`blockIdx` from every thread, see that execution order is not
   guaranteed.
3. **[02_vector_add](02_vector_add/vector_add.cu)** — the full round trip:
   host alloc → device alloc → H2D copy → kernel → D2H copy → verify → free.

Each file has exercises in comments at the bottom — do those before moving
to 202.

## Key terms to walk away understanding

- **kernel** — a function launched on the GPU with `<<<grid, block>>>`, run
  by many threads at once
- **grid / block / thread** — grid = blocks, block = threads; threads in the
  same block can share `__shared__` memory and `__syncthreads()`
- **warp** — 32 threads that execute in lockstep on real hardware; block
  sizes that aren't multiples of 32 waste lanes
- **host vs. device** — host = CPU + its RAM, device = GPU + its VRAM; they
  have separate address spaces, hence `cudaMemcpy`
- **occupancy** — how many warps are actually resident on an SM at once vs.
  the hardware max (you'll measure this in 303)
