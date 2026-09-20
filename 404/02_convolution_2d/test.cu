#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

#define WIDTH 447
#define HEIGHT 447
#define RADIUS 81

__host__ __device__ int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

// ============================================================
// CPU
// ============================================================

void box_blur_cpu(const unsigned char* input, unsigned char* output, int width, int height,
                  int radius) {
    int kernel_area = (2 * radius + 1) * (2 * radius + 1);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int accumulator = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int neighbor_row = clamp(row + dy, 0, height - 1);

                    int neighbor_col = clamp(col + dx, 0, width - 1);

                    accumulator += input[neighbor_row * width + neighbor_col];
                }
            }

            output[row * width + col] = accumulator / kernel_area;
        }
    }
}

// ============================================================
// CUDA - Naive
// ============================================================

__global__ void box_blur_naive(unsigned char* input, unsigned char* output, int width, int height,
                               int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    int accumulator = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int neighbor_row = clamp(row + dy, 0, height - 1);

            int neighbor_col = clamp(col + dx, 0, width - 1);

            accumulator += input[neighbor_row * width + neighbor_col];
        }
    }

    int kernel_area = (2 * radius + 1) * (2 * radius + 1);

    output[row * width + col] = accumulator / kernel_area;
}

// ============================================================
// CUDA - Shared Memory
// ============================================================

__global__ void box_blur_shared(unsigned char* input, unsigned char* output, int width, int height,
                                int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int shared_width = blockDim.x + 2 * radius;

    int shared_height = blockDim.y + 2 * radius;

    extern __shared__ unsigned char shared[];

    int local_col = threadIdx.x + radius;

    int local_row = threadIdx.y + radius;

    // Global memory -> Shared memory
    for (int y = threadIdx.y; y < shared_height; y += blockDim.y) {
        for (int x = threadIdx.x; x < shared_width; x += blockDim.x) {
            int global_col = blockIdx.x * blockDim.x + x - radius;

            int global_row = blockIdx.y * blockDim.y + y - radius;

            global_col = clamp(global_col, 0, width - 1);

            global_row = clamp(global_row, 0, height - 1);

            shared[y * shared_width + x] = input[global_row * width + global_col];
        }
    }

    // Bütün thread'lerin shared memory yüklemesini
    // bitirmesini bekle.
    __syncthreads();

    if (row >= height || col >= width) return;

    int accumulator = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int shared_row = local_row + dy;

            int shared_col = local_col + dx;

            accumulator += shared[shared_row * shared_width + shared_col];
        }
    }

    int kernel_area = (2 * radius + 1) * (2 * radius + 1);

    output[row * width + col] = accumulator / kernel_area;
}

// ============================================================
// Main
// ============================================================

int main() {
    // --------------------------------------------------------
    // Input
    // --------------------------------------------------------

    std::ifstream file_in("lena_gray.raw", std::ios::binary);

    if (!file_in) {
        std::cerr << "Could not open lena_gray.raw\n";
        return 1;
    }

    const size_t image_bytes = WIDTH * HEIGHT * sizeof(unsigned char);

    std::vector<unsigned char> h_input(WIDTH * HEIGHT);

    file_in.read(reinterpret_cast<char*>(h_input.data()), image_bytes);

    // CPU sonucu
    std::vector<unsigned char> h_cpu_output(WIDTH * HEIGHT);

    // GPU sonucu
    std::vector<unsigned char> h_gpu_output(WIDTH * HEIGHT);

    // --------------------------------------------------------
    // CPU
    // --------------------------------------------------------

    auto cpu_start = std::chrono::high_resolution_clock::now();

    box_blur_cpu(h_input.data(), h_cpu_output.data(), WIDTH, HEIGHT, RADIUS);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));

    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), image_bytes, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // Kernel configuration
    // --------------------------------------------------------

    dim3 block(16, 16);

    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    int shared_width = block.x + 2 * RADIUS;

    int shared_height = block.y + 2 * RADIUS;

    size_t shared_bytes = shared_width * shared_height * sizeof(unsigned char);

    // --------------------------------------------------------
    // CUDA Events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float naive_ms = 0.0f;
    float shared_ms = 0.0f;

    // --------------------------------------------------------
    // Naive CUDA
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemset(d_output, 0, image_bytes));

    CUDA_CHECK(cudaEventRecord(start));

    box_blur_naive<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT, RADIUS);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    // --------------------------------------------------------
    // Shared Memory CUDA
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemset(d_output, 0, image_bytes));

    CUDA_CHECK(cudaEventRecord(start));

    box_blur_shared<<<grid, block, shared_bytes>>>(d_input, d_output, WIDTH, HEIGHT, RADIUS);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&shared_ms, start, stop));

    // --------------------------------------------------------
    // GPU result -> CPU
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(h_gpu_output.data(), d_output, image_bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Compare CPU and GPU result
    // --------------------------------------------------------

    size_t mismatch_count = 0;

    for (size_t i = 0; i < h_cpu_output.size(); ++i) {
        if (h_cpu_output[i] != h_gpu_output[i]) {
            ++mismatch_count;
        }
    }

    // --------------------------------------------------------
    // Results
    // --------------------------------------------------------

    std::cout << "\n";
    std::cout << "====================================\n";
    std::cout << "       Box Blur Benchmark\n";
    std::cout << "====================================\n";

    std::cout << "Image           : " << WIDTH << " x " << HEIGHT << "\n";

    std::cout << "Radius          : " << RADIUS << "\n";

    std::cout << "Kernel size     : " << (2 * RADIUS + 1) << " x " << (2 * RADIUS + 1) << "\n";

    std::cout << "Block           : " << block.x << " x " << block.y << "\n";

    std::cout << "\n";

    std::cout << "CPU time        : " << cpu_ms << " ms\n";

    std::cout << "Naive CUDA time : " << naive_ms << " ms\n";

    std::cout << "Shared CUDA time: " << shared_ms << " ms\n";

    std::cout << "\n";

    std::cout << "CPU / Naive     : " << cpu_ms / naive_ms << "x\n";

    std::cout << "CPU / Shared    : " << cpu_ms / shared_ms << "x\n";

    std::cout << "Naive / Shared  : " << naive_ms / shared_ms << "x\n";

    std::cout << "\n";

    std::cout << "CPU vs GPU mismatches: " << mismatch_count << "\n";

    // --------------------------------------------------------
    // Save shared-memory result
    // --------------------------------------------------------

    std::ofstream file_out("blurred_27.raw", std::ios::binary);

    if (!file_out) {
        std::cerr << "Could not create blurred_27.raw\n";
        return 1;
    }

    file_out.write(reinterpret_cast<const char*>(h_gpu_output.data()), h_gpu_output.size());

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}