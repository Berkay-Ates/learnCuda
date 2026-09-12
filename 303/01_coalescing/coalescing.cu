#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define N (1 << 22)
#define STRIDE 32
#define BLOCK_SIZE 256

__global__ void strided_copy(float* in, float* out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int scattered = (idx * stride) % n;
        out[idx] = in[scattered];
    }
}

__global__ void coalesced_copy(float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = in[idx];
    }
}

int main() {
    std::vector<float> h_data(N);
    float result;
    float milliseconds;

    cudaEvent_t start, stop;
    cudaEvent_t start_strided, stop_strided;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start_strided));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&stop_strided));

    for (size_t i = 0; i < N; i++) {
        h_data[i] = 1.0;
    }

    float* d_data;
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaMalloc(&d_result, sizeof(float) * N);

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaEventRecord(start));
    coalesced_copy<<<grid_size, BLOCK_SIZE>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventRecord(start_strided));
    strided_copy<<<grid_size, BLOCK_SIZE>>>(d_data, d_result, N, STRIDE);
    CUDA_CHECK(cudaEventRecord(stop_strided));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Total Elapsed Time for Coalesced Cuda Kernel: %f\n", milliseconds);
    double gb_per_sec = (2.0 * N * sizeof(float) / 1e9) / (milliseconds / 1000.0);
    printf("Coalesced bandwidth: %.1f GB/s\n", gb_per_sec);

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_strided, stop_strided));
    printf("Total Elapsed Time for Strided Cuda Kernel: %f\n", milliseconds);
    gb_per_sec = (2.0 * N * sizeof(float) / 1e9) / (milliseconds / 1000.0);
    printf("Coalesced bandwidth: %.1f GB/s\n", gb_per_sec);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    return 0;
}