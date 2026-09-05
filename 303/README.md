# 303 — Optimization & Parallel Patterns

Goal: stop guessing and start measuring. This is where profiling tools
(`nsys`, `ncu`) enter the workflow.

## Projects (do in order)

1. **reduction** — sum a large array on the GPU three ways and time each:
   - naive: one thread does interleaved-addressing tree reduction with a lot
     of warp divergence
   - improved: sequential addressing to remove divergence, avoid shared
     memory bank conflicts
   - warp-shuffle: use `__shfl_down_sync` to reduce within a warp without
     touching shared memory at all for the last 32 elements

2. **coalescing_demo** — write two kernels that access a large array: one
   with coalesced access (`data[threadIdx.x + blockIdx.x*blockDim.x]`), one
   with a deliberately strided/transposed access pattern. Time both and
   compute effective bandwidth (bytes moved / time) — compare to the peak
   bandwidth number from your `device_query` output.

3. **histogram** — bin a large array of random values using
   `atomicAdd`. Note the throughput hit from atomic contention versus
   vector_add's throughput, then look at using shared-memory sub-histograms
   per block combined into global memory at the end.

## Profiling

```bash
nsys profile ./your_binary       # timeline: kernel/copy overlap, gaps
ncu ./your_binary                 # per-kernel: occupancy, memory throughput, stalls
```

If `nsys`/`ncu` aren't installed with `nvidia-cuda-toolkit`, they ship
separately as Nsight Systems / Nsight Compute — install only if you want to
go deeper here; `cudaEvent` timing (from 101's exercises) is enough to see
the relative differences.

## Concepts you should be able to explain by the end

- Coalesced vs. strided access and why it changes effective bandwidth
- Warp divergence and why `if (threadIdx.x % 2 == 0)`-style branching inside
  a warp is expensive
- Why atomics serialize and how sub-histograms reduce contention
