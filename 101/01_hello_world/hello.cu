#include <cstdio>

#include "../../common/cuda_check.cuh"

__global__ void show3D() {
    printf("thread(%d,%d,%d) in block(%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main() {
    dim3 block(2, 2, 1);
    dim3 grid(2, 2, 1);
    show3D<<<grid, block>>>();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}