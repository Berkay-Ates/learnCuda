#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define N 1024
#define TILE_WIDTH 16

__global__ void matrix_mul_tiled(float* v1, float* v2, float* output, size_t size) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float accumulator = 0;
    int numTiles = (size + TILE_WIDTH - 1) / TILE_WIDTH;

    for (size_t t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < size && aCol < size) ? v1[row * size + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < size && col < size) ? v2[bRow * size + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < size && col < size) {
        output[row * size + col] = accumulator;
    }
}

int main() {
    std::vector<float> v1(N * N);
    std::vector<float> v2(N * N);
    std::vector<float> result(N * N);

    for (size_t i = 0; i < N * N; i++) {
        v1[i] = static_cast<float>(i % 100) * 0.01f;
        v2[i] = static_cast<float>((i + 37) % 100) * 0.01f;
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
            float diff = fabs(expected - result[i * N + j]);
            if (diff > 1e-2f) {  // pick a tolerance appropriate for your value scale
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