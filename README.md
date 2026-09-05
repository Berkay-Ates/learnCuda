# Learn CUDA — 101 to 505

A hands-on path from "what is a kernel" to a small GPU-accelerated project,
using an RTX 3060 (6GB) as the target device.

## Setup (do this first)

```bash
sudo apt update && sudo apt install -y nvidia-cuda-toolkit
nvcc --version
```

Every project has its own `Makefile`. From inside a project folder:

```bash
make        # build
make run    # build + run
make clean  # remove the binary
```

Shared code (like the `CUDA_CHECK` error-checking macro) lives in
[common/cuda_check.cuh](common/cuda_check.cuh) and is included via
`-I../../common` in each Makefile.

## Path

| Folder | Theme | Status |
|---|---|---|
| [101](101/README.md) | Fundamentals: thread hierarchy, host/device memory, error checking | ✅ starter code written |
| [202](202/README.md) | Memory model: 2D indexing, naive vs. shared-memory tiled matmul | 📝 objectives only |
| [303](303/README.md) | Optimization: reduction, coalescing, atomics, profiling | 📝 objectives only |
| [404](404/README.md) | Concurrency: streams, unified memory, 2D shared-memory convolution | 📝 objectives only |
| [505](505/README.md) | Capstone: pick one project combining everything above | 📝 objectives only |

Work through folders in order — each one leans on habits (error checking,
launch config reasoning, shared memory) built in the previous one.
