// Shared CUDA error-checking macro used across every project in this course.
// Wrap any CUDA runtime API call in CUDA_CHECK(...) and it will print the
// file/line and the driver's own error string, then abort, instead of
// silently corrupting results.
//
// Usage:
//   CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
//   CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
//
// After launching a kernel, check for launch errors like this:
//   my_kernel<<<blocks, threads>>>(...);
//   CUDA_CHECK(cudaGetLastError());      // catches bad launch config
//   CUDA_CHECK(cudaDeviceSynchronize()); // catches errors that occur during execution

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)
