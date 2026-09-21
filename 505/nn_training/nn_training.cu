#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "../../common/cuda_check.cuh"

#define BATCH_SIZE 1024
#define INPUT_DIM 2
#define HIDDEN_DIM 8

#define LEARNING_RATE 0.5f
#define EPOCHS 3000

__global__ void sgd_update(float* W, const float* dW, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        W[i] -= lr * dW[i];
    }
}

__global__ void compute_dW1_db1(const float* X, const float* dZ1, float* dW1, float* db1,
                                int batch_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // hangi gizli nöron
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // hangi girdi özelliği

    if (j < INPUT_DIM && k < HIDDEN_DIM) {
        float sum_w = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum_w += X[i * INPUT_DIM + j] * dZ1[i * HIDDEN_DIM + k];
        }
        dW1[j * HIDDEN_DIM + k] = sum_w / batch_size;

        if (j == 0) {  // db1, j'den bağımsız — sadece bir kez hesapla
            float sum_b = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                sum_b += dZ1[i * HIDDEN_DIM + k];
            }
            db1[k] = sum_b / batch_size;
        }
    }
}

__global__ void compute_dZ1(const float* dZ2, const float* W2, const float* Z1, float* dZ1,
                            int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch içindeki örnek
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // gizli nöron

    if (row < batch_size && col < HIDDEN_DIM) {
        float dA1 = dZ2[row] * W2[col];
        float relu_mask = (Z1[row * HIDDEN_DIM + col] > 0.0f) ? 1.0f : 0.0f;
        dZ1[row * HIDDEN_DIM + col] = dA1 * relu_mask;
    }
}

__global__ void compute_dW2_db2(const float* A1, const float* dZ2, float* dW2, float* db2,
                                int batch_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < HIDDEN_DIM) {
        float sum_w = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum_w += A1[i * HIDDEN_DIM + k] * dZ2[i];
        }
        dW2[k] = sum_w / batch_size;

        if (k == 0) {
            float sum_b = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                sum_b += dZ2[i];
            }
            db2[0] = sum_b / batch_size;
        }
    }
}

__global__ void compute_dZ2(const float* A2, const float* Y, float* dZ2, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        dZ2[i] = A2[i] - Y[i];
    }
}

__global__ void forward_output(const float* A1, const float* W2, const float* b2, float* Z2,
                               float* A2, int batch_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size) {
        float sum = b2[0];
        for (int k = 0; k < HIDDEN_DIM; k++) {
            sum += A1[row * HIDDEN_DIM + k] * W2[k];
        }
        Z2[row] = sum;
        A2[row] = 1.0f / (1.0f + expf(-sum));
    }
}

__global__ void forward_hidden(const float* X, const float* W1, const float* b1, float* Z1,
                               float* A1, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // hangi örnek (batch içinde)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // hangi gizli nöron

    if (row < batch_size && col < HIDDEN_DIM) {
        float sum = b1[col];
        for (int k = 0; k < INPUT_DIM; k++) {
            sum += X[row * INPUT_DIM + k] * W1[k * HIDDEN_DIM + col];
        }
        Z1[row * HIDDEN_DIM + col] = sum;
        A1[row * HIDDEN_DIM + col] = fmaxf(0.0f, sum);  // ReLU
    }
}

int main() {
    std::vector<float> h_X(BATCH_SIZE * INPUT_DIM);
    std::vector<float> h_Y(BATCH_SIZE);

    float corners[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float labels[4] = {0, 1, 1, 0};

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> corner_dist(0, 3);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    for (int i = 0; i < BATCH_SIZE; ++i) {
        int c = corner_dist(gen);

        h_X[i * INPUT_DIM + 0] = corners[c][0] + noise(gen);
        h_X[i * INPUT_DIM + 1] = corners[c][1] + noise(gen);

        h_Y[i] = labels[c];
    }

    std::vector<float> h_W1(INPUT_DIM * HIDDEN_DIM);
    std::vector<float> h_b1(HIDDEN_DIM, 0.0f);
    std::vector<float> h_W2(HIDDEN_DIM * 1);
    std::vector<float> h_b2(1, 0.0f);

    std::vector<float> h_A2(BATCH_SIZE);

    std::uniform_real_distribution<float> weight_dist(-0.5f, 0.5f);
    for (auto& w : h_W1) w = weight_dist(gen);
    for (auto& w : h_W2) w = weight_dist(gen);

    float* d_X;
    float* d_Y;
    float* d_W1;
    float* d_b1;
    float* d_W2;
    float* d_b2;

    float* d_Z1;
    float* d_A1;

    // Memory Malloc
    CUDA_CHECK(cudaMalloc(&d_X, BATCH_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, BATCH_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W1, INPUT_DIM * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, HIDDEN_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W2, HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Z1, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A1, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));

    // host to device copy
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), BATCH_SIZE * INPUT_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_Y, h_Y.data(), BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), INPUT_DIM * HIDDEN_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), sizeof(float), cudaMemcpyHostToDevice));
    // Kernel Launch

    dim3 block(8, 32);
    dim3 grid((HIDDEN_DIM + block.x - 1) / block.x, (BATCH_SIZE + block.y - 1) / block.y);

    float* d_Z2;
    float* d_A2;
    float* d_dZ2;

    CUDA_CHECK(cudaMalloc(&d_Z2, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A2, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dZ2, BATCH_SIZE * sizeof(float)));

    dim3 block_output(256);
    dim3 grid_output((BATCH_SIZE + block_output.x - 1) / block_output.x);

    dim3 block_dz2(256);
    dim3 grid_dz2((BATCH_SIZE + block_dz2.x - 1) / block_dz2.x);

    float* d_dW2;
    float* d_db2;

    CUDA_CHECK(cudaMalloc(&d_dW2, HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, sizeof(float)));

    dim3 block_dw2(HIDDEN_DIM);
    dim3 grid_dw2(1);

    float* d_dZ1;

    CUDA_CHECK(cudaMalloc(&d_dZ1, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));

    float* d_dW1;
    float* d_db1;

    CUDA_CHECK(cudaMalloc(&d_dW1, INPUT_DIM * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, HIDDEN_DIM * sizeof(float)));
    dim3 block_dw1(HIDDEN_DIM, INPUT_DIM);
    dim3 grid_dw1(1, 1);

    dim3 block_update(256);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        forward_hidden<<<grid, block>>>(d_X, d_W1, d_b1, d_Z1, d_A1, BATCH_SIZE);

        forward_output<<<grid_output, block_output>>>(d_A1, d_W2, d_b2, d_Z2, d_A2, BATCH_SIZE);

        compute_dZ2<<<grid_dz2, block_dz2>>>(d_A2, d_Y, d_dZ2, BATCH_SIZE);

        compute_dW2_db2<<<grid_dw2, block_dw2>>>(d_A1, d_dZ2, d_dW2, d_db2, BATCH_SIZE);

        compute_dZ1<<<grid, block>>>(d_dZ2, d_W2, d_Z1, d_dZ1, BATCH_SIZE);

        compute_dW1_db1<<<grid_dw1, block_dw1>>>(d_X, d_dZ1, d_dW1, d_db1, BATCH_SIZE);

        sgd_update<<<1, block_update>>>(d_W1, d_dW1, LEARNING_RATE, INPUT_DIM * HIDDEN_DIM);

        sgd_update<<<1, block_update>>>(d_b1, d_db1, LEARNING_RATE, HIDDEN_DIM);

        sgd_update<<<1, block_update>>>(d_W2, d_dW2, LEARNING_RATE, HIDDEN_DIM);

        sgd_update<<<1, block_update>>>(d_b2, d_db2, LEARNING_RATE, 1);

        if (epoch % 200 == 0) {
            CUDA_CHECK(
                cudaMemcpy(h_A2.data(), d_A2, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            float loss = 0.0f;

            for (int i = 0; i < BATCH_SIZE; ++i) {
                float a = h_A2[i];
                float y = h_Y[i];

                loss += -(y * logf(a + 1e-7f) + (1.0f - y) * logf(1.0f - a + 1e-7f));
            }

            loss /= BATCH_SIZE;

            std::cout << "Epoch " << epoch << " | Loss: " << loss << '\n';
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int correct = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        int predicted = h_A2[i] > 0.5f ? 1 : 0;
        int actual = h_Y[i] > 0.5f ? 1 : 0;
        if (predicted == actual) correct++;
    }
    std::cout << "\nFinal accuracy: " << (100.0f * correct / BATCH_SIZE) << "%\n";

    // Free memory
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_Z1));
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_Z2));
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_dZ2));
    CUDA_CHECK(cudaFree(d_db2));
    CUDA_CHECK(cudaFree(d_dW2));
    CUDA_CHECK(cudaFree(d_dZ1));
    CUDA_CHECK(cudaFree(d_dW1));
    CUDA_CHECK(cudaFree(d_db1));

    return 0;
}