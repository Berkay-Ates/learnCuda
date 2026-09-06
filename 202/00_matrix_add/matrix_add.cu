#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define N 16

__global__ void matrix_add(float* v1, float* v2, float* output, size_t size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < size && col < size) {
        size_t index = row * size + col;
        output[index] = v1[index] + v2[index];
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

    matrix_add<<<gridDim, blockDim>>>(d_v1, d_v2, d_result, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), d_result, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    size_t mismatch_count = 0;
    for (size_t i = 0; i < N * N; i++) {
        float const expected = v1[i] + v2[i];
        if (expected != result[i]) {
            mismatch_count++;
        }
    }
    printf("Mismatch count: %zu\n", mismatch_count);

    CUDA_CHECK(cudaFree(d_v1));
    CUDA_CHECK(cudaFree(d_v2));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}