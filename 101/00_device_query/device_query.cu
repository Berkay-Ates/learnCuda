#include <cstdio>

#include "../../common/cuda_check.cuh"

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::printf("cudaGetDeviceCount: %d\n", deviceCount);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("prop.name: %s\n", prop.name);
    std::printf("SM Count: %d\n", prop.multiProcessorCount);
    std::printf("Wrap Size: %d\n", prop.warpSize);
    std::printf("Max Thread Per Block: %d\n", prop.maxThreadsPerBlock);
    std::printf("Max Thread Per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("Max Thread can run on this GPU (Theoretically): %d\n",
                prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);

    printf("Shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Shared mem per block: %zu kilobytes\n", prop.sharedMemPerBlock / 1024);

    return 0;
}