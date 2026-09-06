#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define N 1024         // büyük tutuyoruz ki fark net görünsün
#define TILE_WIDTH 16  // blockDim de bununla eşleşecek (16x16 = 256 thread/blok)

// ---------------- Naive GPU ----------------
__global__ void matrix_mul_naive(const float* v1, const float* v2, float* output, size_t size) {
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

// ---------------- Tiled GPU ----------------
__global__ void matrix_mul_tiled(const float* v1, const float* v2, float* output, size_t size) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float accumulator = 0;

    int numTiles = (size + TILE_WIDTH - 1) / TILE_WIDTH;  // k boyunca kaç tile adımı var

    for (int t = 0; t < numTiles; t++) {
        // --- 1. cooperatif yükleme: her thread SADECE kendi payına düşen tek elemanı yükler ---
        int aCol = t * TILE_WIDTH + threadIdx.x;  // A'dan okunacak sütun (k boyutunda kayan)
        int bRow = t * TILE_WIDTH + threadIdx.y;  // B'den okunacak satır (k boyutunda kayan)

        tileA[threadIdx.y][threadIdx.x] =
            (row < size && aCol < size) ? v1[row * size + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < size && col < size) ? v2[bRow * size + col] : 0.0f;

        __syncthreads();  // tile tamamen dolmadan kimse okumasın

        // --- 2. shared memory'deki tile'ı kullanarak kısmi toplamı biriktir ---
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();  // herkes okumayı bitirmeden bir sonraki adım tile'ı ezmesin
    }

    if (row < size && col < size) {
        output[row * size + col] = accumulator;
    }
}

// ---------------- CPU reference ----------------
void matrix_mul_cpu(const std::vector<float>& v1, const std::vector<float>& v2,
                    std::vector<float>& output, size_t size) {
    for (size_t row = 0; row < size; row++) {
        for (size_t col = 0; col < size; col++) {
            float accumulator = 0;
            for (size_t k = 0; k < size; k++) {
                accumulator += v1[row * size + k] * v2[k * size + col];
            }
            output[row * size + col] = accumulator;
        }
    }
}

// ---------------- Yardımcı: GPU kernel'i zamanla + doğrula ----------------
float run_gpu_kernel(void (*kernel)(const float*, const float*, float*, size_t), const char* label,
                     float* d_v1, float* d_v2, float* d_result, std::vector<float>& gpu_result,
                     const std::vector<float>& cpu_result, dim3 gridDim, dim3 blockDim) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<gridDim, blockDim>>>(d_v1, d_v2, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(
        cudaMemcpy(gpu_result.data(), d_result, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    size_t mismatch_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < N * N; i++) {
        float diff = std::fabs(cpu_result[i] - gpu_result[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-1f) mismatch_count++;  // N büyüdükçe tolerans da büyütülmeli
    }

    printf("%-12s time: %8.3f ms | mismatch: %zu | max diff: %e\n", label, ms, mismatch_count,
           max_diff);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    std::vector<float> v1(N * N);
    std::vector<float> v2(N * N);
    std::vector<float> cpu_result(N * N);
    std::vector<float> gpu_result(N * N);

    for (size_t i = 0; i < N * N; i++) {
        v1[i] = static_cast<float>(i % 100) * 0.01f;
        v2[i] = static_cast<float>((i + 37) % 100) * 0.01f;
    }

    // ---------- CPU ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrix_mul_cpu(v1, v2, cpu_result, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ---------- GPU setup (tek sefer, ikisi de kullanacak) ----------
    float* d_v1 = NULL;
    float* d_v2 = NULL;
    float* d_result = NULL;

    CUDA_CHECK(cudaMalloc(&d_v1, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v2, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_v1, v1.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v2, v2.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    printf("N = %d, TILE_WIDTH = %d\n", N, TILE_WIDTH);
    printf("CPU time:      %8.3f ms\n", cpu_ms);

    float naive_ms = run_gpu_kernel(matrix_mul_naive, "Naive GPU", d_v1, d_v2, d_result, gpu_result,
                                    cpu_result, gridDim, blockDim);

    float tiled_ms = run_gpu_kernel(matrix_mul_tiled, "Tiled GPU", d_v1, d_v2, d_result, gpu_result,
                                    cpu_result, gridDim, blockDim);

    printf("\nSpeedup naive vs CPU:  %.2fx\n", cpu_ms / naive_ms);
    printf("Speedup tiled vs CPU:  %.2fx\n", cpu_ms / tiled_ms);
    printf("Speedup tiled vs naive: %.2fx\n", naive_ms / tiled_ms);

    CUDA_CHECK(cudaFree(d_v1));
    CUDA_CHECK(cudaFree(d_v2));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}