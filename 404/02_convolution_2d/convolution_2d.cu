#include <fstream>
#include <iostream>
#include <vector>

#include "../../common/cuda_check.cuh"

#define WIDTH 447
#define HEIGHT 447
#define RADIUS 27

__host__ __device__ int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

__global__ void box_blur_shared(unsigned char* input, unsigned char* output, int width, int height,
                                int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;

    int shared_width = BLOCK_X + 2 * radius;
    int shared_height = BLOCK_Y + 2 * radius;

    extern __shared__ unsigned char shared[];

    // Shared memory içindeki başlangıç noktası
    // thread'in global koordinatına karşılık gelen merkez pixel
    int local_col = threadIdx.x + radius;
    int local_row = threadIdx.y + radius;

    // Öncelikle merkez bölgeyi yükle
    for (int y = threadIdx.y; y < shared_height; y += blockDim.y) {
        for (int x = threadIdx.x; x < shared_width; x += blockDim.x) {
            int global_col = blockIdx.x * blockDim.x + x - radius;

            int global_row = blockIdx.y * blockDim.y + y - radius;

            global_col = clamp(global_col, 0, width - 1);
            global_row = clamp(global_row, 0, height - 1);

            shared[y * shared_width + x] = input[global_row * width + global_col];
        }
    }

    // Bütün block shared memory'yi doldurmadan
    // hiçbir thread hesaplamaya başlamasın.
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

int main() {
    std::ifstream file_in("lena_gray.raw", std::ios::binary);
    std::vector<unsigned char> h_input(WIDTH * HEIGHT);
    file_in.read(reinterpret_cast<char*>(h_input.data()), h_input.size());

    std::vector<unsigned char> h_output(WIDTH * HEIGHT);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), WIDTH * HEIGHT * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, WIDTH * HEIGHT * sizeof(unsigned char)));

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    int shared_width = block.x + 2 * RADIUS;
    int shared_height = block.y + 2 * RADIUS;
    size_t shared_bytes = shared_width * shared_height * sizeof(unsigned char);

    // box_blur_naive<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT, RADIUS);
    box_blur_shared<<<grid, block, shared_bytes>>>(d_input, d_output, WIDTH, HEIGHT, RADIUS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, WIDTH * HEIGHT * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::ofstream file_out("blurred_27.raw", std::ios::binary);
    file_out.write(reinterpret_cast<const char*>(h_output.data()), h_output.size());

    return 0;
}