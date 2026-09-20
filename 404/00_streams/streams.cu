#include <cstdio>
#include <random>

#include "../../common/cuda_check.cuh"

#define N (1 << 24)
#define BLOCK_SIZE 256
#define NUM_STREAMS 64

__global__ void heavy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int r = 0; r < 250; r++) {
            val = sinf(val) * cosf(val) + 1.0f;
        }
        data[idx] = val;
    }
}

int main() {
    // -------------------------
    // Host data
    // -------------------------
    float* h_data;
    float* h_result;

    cudaStream_t streams[NUM_STREAMS];

    CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_result, N * sizeof(float)));

    std::mt19937 gen(42);  // fixed seed
    std::uniform_real_distribution<float> dist(0, 1.0f);

    for (size_t i = 0; i < N; i++) {
        h_data[i] = dist(gen);
    }

    for (size_t i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // -------------------------
    // Device memory
    // -------------------------
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(float) * N));

    // -------------------------
    // Timing
    // -------------------------
    cudaEvent_t start;
    cudaEvent_t start2;
    cudaEvent_t stop;
    cudaEvent_t stop2;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&stop2));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, sizeof(float) * N, cudaMemcpyHostToDevice, 0));

    // -------------------------
    // Kernel configuration
    // -------------------------
    int chunk_size = N / NUM_STREAMS;
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    heavy_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, N);
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_data, sizeof(float) * N, cudaMemcpyDeviceToHost, 0));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    std::vector<float> h_result_baseline(N);
    memcpy(h_result_baseline.data(), h_result, N * sizeof(float));

    float milliseconds;

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Elapsed time: %.3f ms\n", milliseconds);

    // -------------------------
    // Streamed Data Processing
    // -------------------------

    CUDA_CHECK(cudaEventRecord(start2));

    for (size_t i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;

        cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_size * sizeof(float),
                        cudaMemcpyHostToDevice, streams[i]);

        int chunk_grid = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        heavy_kernel<<<chunk_grid, BLOCK_SIZE, 0, streams[i]>>>(d_data + offset, chunk_size);

        cudaMemcpyAsync(h_result + offset, d_data + offset, chunk_size * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop2));
    CUDA_CHECK(cudaEventSynchronize(stop2));

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start2, stop2));

    printf("Streamed elapsed time: %.3f ms\n", milliseconds);

    // ... streamed kısmı çalıştıktan sonra:
    int mismatch = 0;
    for (int i = 0; i < N; i++) {
        if (h_result_baseline[i] != h_result[i]) mismatch++;
    }
    printf("Mismatch: %d\n", mismatch);

    // -------------------------
    // Cleanup
    // -------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(stop2));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFreeHost(h_result));

    for (size_t i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}