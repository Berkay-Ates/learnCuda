#include <cstdio>
#include <random>
#include <vector>

#include "../../common/cuda_check.cuh"

#define N (1 << 24)
#define NUM_BINS 256
#define BLOCK_SIZE 256

__global__ void histogram_shared(int* data, int* histogram, int n) {
    __shared__ int local_hist[NUM_BINS];

    // 1. Shared histogram'u sıfırla
    local_hist[threadIdx.x] = 0;

    // 2. Her thread'in sıfırlaması bitsin
    __syncthreads();

    // 3. Global data'dan kendi elemanını al
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 4. Artık global histogram değil,
        //    block'un kendi histogram'una atomic
        atomicAdd(&local_hist[data[idx]], 1);
    }

    // 5. Block içindeki bütün atomic işlemler bitsin
    __syncthreads();

    // 6. Her thread kendi bin'ini global histogram'a aktarır
    atomicAdd(&histogram[threadIdx.x], local_hist[threadIdx.x]);
}

__global__ void histogram_naive(int* data, int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}

int main() {
    // -------------------------
    // 1. Host data
    // -------------------------
    std::vector<int> h_data(N);

    std::mt19937 gen(42);  // fixed seed
    std::uniform_int_distribution<int> dist(0, NUM_BINS - 1);

    for (int& value : h_data) {
        value = dist(gen);
    }

    // -------------------------
    // 2. Device memory
    // -------------------------
    int* d_data;
    int* d_histogram;

    CUDA_CHECK(cudaMalloc(&d_data, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_histogram, sizeof(int) * NUM_BINS));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, sizeof(int) * NUM_BINS));

    // -------------------------
    // 3. Kernel configuration
    // -------------------------
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // -------------------------
    // 4. Timing
    // -------------------------
    cudaEvent_t start;
    cudaEvent_t start_shared;
    cudaEvent_t stop;
    cudaEvent_t stop_shared;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start_shared));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&stop_shared));

    CUDA_CHECK(cudaEventRecord(start));
    histogram_naive<<<grid_size, BLOCK_SIZE>>>(d_data, d_histogram, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventRecord(start_shared));
    histogram_shared<<<grid_size, BLOCK_SIZE>>>(d_data, d_histogram, N);
    CUDA_CHECK(cudaEventRecord(stop_shared));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventSynchronize(stop_shared));

    float milliseconds;

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Naive histogram time: %.3f ms\n", milliseconds);

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_shared, stop_shared));
    printf("Sahared histogram time: %.3f ms\n", milliseconds);

    // -------------------------
    // 5. Copy GPU result
    // -------------------------
    int h_histogram[NUM_BINS] = {0};

    CUDA_CHECK(
        cudaMemcpy(h_histogram, d_histogram, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost));

    // -------------------------
    // 6. CPU reference
    // -------------------------
    int cpu_histogram[NUM_BINS] = {0};

    for (int value : h_data) {
        cpu_histogram[value]++;
    }

    // -------------------------
    // 7. Verify
    // -------------------------
    bool correct = true;

    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_histogram[i] != 2 * cpu_histogram[i]) {
            printf("Mismatch at bin %d: GPU=%d CPU=%d\n", i, h_histogram[i], cpu_histogram[i]);

            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Histogram verification: PASSED\n");
    } else {
        printf("Histogram verification: FAILED\n");
    }

    // -------------------------
    // 8. Cleanup
    // -------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histogram));

    return 0;
}