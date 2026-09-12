#include <cmath>
#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define N (1 << 20)
#define BLOCK_SIZE 256

__global__ void add_all_elements(float* values, float* result) {
    // STEP 1: Shared memory (block'un kendi scratch space'i)
    __shared__ float sdata[BLOCK_SIZE];

    // STEP 2: Index hesapla
    unsigned int tid = threadIdx.x;                          // 0-255 (block içinde)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global

    // STEP 3: Load phase (global → shared memory, COALESCED!)
    sdata[tid] = (i < N) ? values[i] : 0.0f;

    // STEP 4: Birlikte load bitene kadar bekle
    __syncthreads();

    // STEP 5: Tree reduction loop
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        // Sadece aktif thread'ler (tid < stride) işlem yapıyor
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        // HER İTERASYONDA bekle (ÖNEMLI!)
        __syncthreads();
    }

    // STEP 6: Sonuç (sdata[0]) global memory'ye yazdir
    // Ama sadece 1 thread yapsin (aksi halde atomic 256 kez çalışır!)
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    std::vector<float> h_data(N);
    float result;

    for (size_t i = 0; i < N; i++) {
        h_data[i] = 1.0;
    }

    float* d_data;
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    // Kernel Launch
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_all_elements<<<grid_size, BLOCK_SIZE>>>(d_data, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Result: %.0f (expected: %.0f)\n", result, (float)N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}