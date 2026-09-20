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
