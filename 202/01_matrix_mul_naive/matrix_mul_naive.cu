#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define N 16

__global__ void matrix_mul_tiled(float* v1, float* v2, float* output, size_t size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < size && row < size) {
        float accumulator = 0;
        for (int k = 0; k < size; k++) {
            accumulator += v1[row * size + k] * v2[k * size + col];
        }
        output[row * size + col] = accumulator;
    }
}

int main() {
    std::vector<float> v1(N * N);
    std::vector<float> v2(N * N);
    std::vector<float> result(N * N);

    for (size_t i = 0; i < N * N; i++) {
        v1[i] = static_cast<float>(i);
        v2[i] = static_cast<float>(i + 100);
    }

    float* d_v1;
    float* d_v2;
    float* d_result;

    CUDA_CHECK(cudaMalloc(&d_v1, sizeof(float) * N * N));
    CUDA_CHECK(cudaMalloc(&d_v2, sizeof(float) * N * N));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float) * N * N));

    CUDA_CHECK(cudaMemcpy(d_v1, v1.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v2, v2.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);                           // 256 thread/block (common)
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,  // same ceiling-division trick, per axis now
                 (N + blockDim.y - 1) / blockDim.y);

    matrix_mul_tiled<<<gridDim, blockDim>>>(d_v1, d_v2, d_result, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), d_result, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    size_t mismatch_count = 0;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            float expected = 0;
            for (size_t k = 0; k < N; k++) {
                expected += v1[i * N + k] * v2[k * N + j];
            }
            if (expected != result[i * N + j]) {
                mismatch_count++;
            }
        }
    }
    printf("Mismatch count: %zu\n", mismatch_count);

    CUDA_CHECK(cudaFree(d_v1));
    CUDA_CHECK(cudaFree(d_v2));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}