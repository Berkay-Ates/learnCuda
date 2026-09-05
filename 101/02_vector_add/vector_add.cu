#include <cstdio>
#include <vector>

#include "../../common/cuda_check.cuh"

#define SIZE 1000

__global__ void Add_Vectors(float* v1, float* v2, float* output, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = v1[index] + v2[index];
    }
}

int main() {
    std::vector<float> v1(SIZE);
    std::vector<float> v2(SIZE);
    std::vector<float> result(SIZE);

    for (size_t i = 0; i < SIZE; i++) {
        v1[i] = i;
        v2[i] = (i + 100);
    }

    float* d_v1 = NULL;
    float* d_v2 = NULL;
    float* d_result = NULL;

    CUDA_CHECK(cudaMalloc(&d_v1, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v2, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, SIZE * sizeof(float)));

    cudaError_t v1_cpy = cudaMemcpy(d_v1, v1.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t v2_cpy = cudaMemcpy(d_v2, v2.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);

    CUDA_CHECK(v1_cpy);
    CUDA_CHECK(v2_cpy);

    int threadsPerBlock = 256;
    int blocks = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    Add_Vectors<<<blocks, threadsPerBlock>>>(d_v1, d_v2, d_result, SIZE);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t result_cpy =
        cudaMemcpy(result.data(), d_result, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(result_cpy);

    size_t mismatch_count = 0;
    for (size_t i = 0; i < SIZE; i++) {
        float expected = v1[i] + v2[i];
        if (result[i] != expected) {
            printf("Mismatch at %zu: got %f, expected %f\n", i, result[i], expected);
            mismatch_count++;
        }
    }
    printf("Mismatch Count: %zu\n", mismatch_count);

    CUDA_CHECK(cudaFree(d_v1));
    CUDA_CHECK(cudaFree(d_v2));
    CUDA_CHECK(cudaFree(d_result));

    v1.clear();
    v2.clear();
    result.clear();

    return 0;
}