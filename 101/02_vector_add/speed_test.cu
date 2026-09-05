#include <chrono>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define SIZE (1 << 24)  // ~16.7 million elements (~64MB per array) — big enough to see a real gap

__global__ void Add_Vectors_GPU(const float* v1, const float* v2, float* output, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = v1[index] + v2[index];
    }
}

void Add_Vectors_CPU(const std::vector<float>& v1, const std::vector<float>& v2,
                     std::vector<float>& output) {
    for (size_t i = 0; i < v1.size(); i++) {
        output[i] = v1[i] + v2[i];
    }
}

int main() {
    std::vector<float> v1(SIZE);
    std::vector<float> v2(SIZE);
    std::vector<float> cpu_result(SIZE);
    std::vector<float> gpu_result(SIZE);

    for (size_t i = 0; i < SIZE; i++) {
        v1[i] = static_cast<float>(i);
        v2[i] = static_cast<float>(i + 100);
    }

    // ---------- CPU timing ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    Add_Vectors_CPU(v1, v2, cpu_result);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ---------- GPU setup ----------
    float* d_v1 = NULL;
    float* d_v2 = NULL;
    float* d_result = NULL;

    CUDA_CHECK(cudaMalloc(&d_v1, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v2, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_v1, v1.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v2, v2.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // ---------- GPU timing (cudaEvent_t — see note below) ----------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Add_Vectors_GPU<<<blocks, threadsPerBlock>>>(d_v1, d_v2, d_result, SIZE);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));  // wait until 'stop' event actually completes

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    CUDA_CHECK(
        cudaMemcpy(gpu_result.data(), d_result, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------- Verify ----------
    size_t mismatch_count = 0;
    for (size_t i = 0; i < SIZE; i++) {
        if (gpu_result[i] != cpu_result[i]) {
            mismatch_count++;
        }
    }

    printf("CPU time: %.3f ms\n", cpu_ms);
    printf("GPU time: %.3f ms (kernel only)\n", gpu_ms);
    printf("Speedup:  %.2fx\n", cpu_ms / gpu_ms);
    printf("Mismatch count: %zu\n", mismatch_count);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_v1));
    CUDA_CHECK(cudaFree(d_v2));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}