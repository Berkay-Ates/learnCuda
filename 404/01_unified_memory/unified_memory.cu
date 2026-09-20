#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define SIZE (1 << 24)

__global__ void Add_Vectors(const float* v1, const float* v2, float* output, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < 1000; i++) {
        if (index < size) {
            output[index] = v1[index] + v2[index];
        }
    }
}

int main() {
    // ============================================================
    // Host memory
    // ============================================================

    std::vector<float> host_v1(SIZE);
    std::vector<float> host_v2(SIZE);

    // Explicit CUDA sonucu
    std::vector<float> explicit_result(SIZE);

    // Unified Memory sonucu
    std::vector<float> unified_result(SIZE);

    for (size_t i = 0; i < SIZE; i++) {
        host_v1[i] = static_cast<float>(i);
        host_v2[i] = static_cast<float>(i + 100);
    }

    int threadsPerBlock = 256;
    int blocks = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // ============================================================
    // 1. Explicit CUDA Memory
    // ============================================================

    float* device_v1 = nullptr;
    float* device_v2 = nullptr;
    float* device_result = nullptr;

    CUDA_CHECK(cudaMalloc(&device_v1, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_v2, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_result, SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(device_v1, host_v1.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(device_v2, host_v2.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t explicit_start;
    cudaEvent_t explicit_stop;

    CUDA_CHECK(cudaEventCreate(&explicit_start));
    CUDA_CHECK(cudaEventCreate(&explicit_stop));

    CUDA_CHECK(cudaEventRecord(explicit_start));

    Add_Vectors<<<blocks, threadsPerBlock>>>(device_v1, device_v2, device_result, SIZE);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(explicit_stop));
    CUDA_CHECK(cudaEventSynchronize(explicit_stop));

    float explicit_time = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&explicit_time, explicit_start, explicit_stop));

    CUDA_CHECK(cudaMemcpy(explicit_result.data(), device_result, SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ============================================================
    // 2. Unified Memory
    // ============================================================

    float* unified_v1 = nullptr;
    float* unified_v2 = nullptr;
    float* unified_result_ptr = nullptr;

    CUDA_CHECK(cudaMallocManaged(&unified_v1, SIZE * sizeof(float)));

    CUDA_CHECK(cudaMallocManaged(&unified_v2, SIZE * sizeof(float)));

    CUDA_CHECK(cudaMallocManaged(&unified_result_ptr, SIZE * sizeof(float)));

    for (size_t i = 0; i < SIZE; i++) {
        unified_v1[i] = static_cast<float>(i);
        unified_v2[i] = static_cast<float>(i + 100);
    }

    CUDA_CHECK(cudaMemPrefetchAsync(unified_v1, SIZE * sizeof(float), 0));

    CUDA_CHECK(cudaMemPrefetchAsync(unified_v2, SIZE * sizeof(float), 0));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t unified_start;
    cudaEvent_t unified_stop;

    CUDA_CHECK(cudaEventCreate(&unified_start));
    CUDA_CHECK(cudaEventCreate(&unified_stop));

    CUDA_CHECK(cudaEventRecord(unified_start));

    Add_Vectors<<<blocks, threadsPerBlock>>>(unified_v1, unified_v2, unified_result_ptr, SIZE);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(unified_stop));
    CUDA_CHECK(cudaEventSynchronize(unified_stop));

    float unified_time = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&unified_time, unified_start, unified_stop));

    // GPU işlemi bittikten sonra CPU tarafından okunabilir.
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < SIZE; i++) {
        unified_result[i] = unified_result_ptr[i];
    }

    // ============================================================
    // 3. Explicit Memory doğruluk kontrolü
    // ============================================================

    size_t explicit_mismatch_count = 0;

    for (size_t i = 0; i < SIZE; i++) {
        float expected = host_v1[i] + host_v2[i];

        if (explicit_result[i] != expected) {
            printf("[Explicit] Mismatch at %zu: got %f, expected %f\n", i, explicit_result[i],
                   expected);

            explicit_mismatch_count++;
        }
    }

    // ============================================================
    // 4. Unified Memory doğruluk kontrolü
    // ============================================================

    size_t unified_mismatch_count = 0;

    for (size_t i = 0; i < SIZE; i++) {
        float expected = host_v1[i] + host_v2[i];

        if (unified_result[i] != expected) {
            printf("[Unified] Mismatch at %zu: got %f, expected %f\n", i, unified_result[i],
                   expected);

            unified_mismatch_count++;
        }
    }

    // ============================================================
    // 5. Results
    // ============================================================

    printf("\n==============================\n");
    printf("Results\n");
    printf("==============================\n");

    printf("Explicit CUDA kernel time : %.4f ms\n", explicit_time);

    printf("Unified Memory kernel time: %.4f ms\n", unified_time);

    printf("Explicit mismatches       : %zu\n", explicit_mismatch_count);

    printf("Unified mismatches        : %zu\n", unified_mismatch_count);

    // ============================================================
    // Cleanup
    // ============================================================

    CUDA_CHECK(cudaFree(device_v1));
    CUDA_CHECK(cudaFree(device_v2));
    CUDA_CHECK(cudaFree(device_result));

    CUDA_CHECK(cudaFree(unified_v1));
    CUDA_CHECK(cudaFree(unified_v2));
    CUDA_CHECK(cudaFree(unified_result_ptr));

    CUDA_CHECK(cudaEventDestroy(explicit_start));
    CUDA_CHECK(cudaEventDestroy(explicit_stop));

    CUDA_CHECK(cudaEventDestroy(unified_start));
    CUDA_CHECK(cudaEventDestroy(unified_stop));

    return 0;
}