# 505 — Capstone

Goal: one project that combines shared-memory tiling, launch-config
reasoning, and (ideally) streams or profiling from 303/404. Pick one.

## Option A — Raw-CUDA NN layer forward pass

`y = ReLU(W @ x + b)` for a batch of inputs, implemented from scratch:
tiled matmul (reuse 202) + a fused bias-add + ReLU kernel (or fuse it into
the matmul epilogue). Ties directly back to 202/303. Good choice if you're
interested in ML systems / why frameworks like PyTorch need custom kernels
at all.

## Option B — Bitonic sort

A sorting network that's parallel-friendly (unlike quicksort/mergesort).
Good choice for practicing `__syncthreads()` discipline and in-place
shared-memory swaps across multiple kernel launches (one per stage).

## Option C — N-body simulation

Naive O(n²) gravity simulation: each thread computes the force on one body
from all others, tiled through shared memory (the classic
"GPU Gems 3 N-body" pattern). Good choice if you like physics/visual output
— pairs well with dumping frames and making a video.

## Option D — Image processing pipeline

Chain grayscale → Gaussian blur → Sobel edge detection as three (or fused)
kernels over a real image, using streams to overlap where possible. Direct
extension of 404's convolution project. Good choice if you want a visibly
"cool" result (before/after images) to show for the project.

## Suggested process regardless of which you pick

1. Write a correct CPU reference implementation first.
2. Write the naive GPU version, verify against the CPU reference.
3. Profile it, find the bottleneck (`nsys`/`ncu` or `cudaEvent` timing).
4. Optimize the specific bottleneck you measured — not a guess.
5. Write up before/after numbers.
