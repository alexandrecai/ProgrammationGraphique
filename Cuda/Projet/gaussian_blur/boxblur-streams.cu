#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

__global__ void grayscale_boxblur_shared(unsigned char* rgb, unsigned char* s, std::size_t cols, std::size_t rows)
{
    auto i = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    auto j = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    auto w = blockDim.x;
    auto h = blockDim.y;

    extern __shared__ unsigned char sh[];

    if (i < cols && j < rows) {
        sh[lj * w + li] = (307 * rgb[3 * (j * cols + i)] + 604 * rgb[3 * (j * cols + i) + 1] + 113 * rgb[3 * (j * cols + i) + 2]) >> 10;
    }

    __syncthreads();

    if (i < cols - 1 && j < rows - 1 && li > 0 && li < (w - 1) && lj > 0 && lj < (h - 1))
    {
        auto total = sh[((lj - 1) * w + li - 1)] + sh[((lj - 1) * w + li)] + sh[((lj - 1) * w + li + 1)]
                     + sh[(lj * w + li - 1)] + sh[(lj * w + li)] + sh[(lj * w + li + 1)]
                     + sh[((lj + 1) * w + li - 1)] + sh[((lj + 1) * w + li)] + sh[((lj + 1) * w + li + 1)];

        auto res = total / 9;
        s[j * cols + i] = res;
    }
}

int main() {
    // Lecture de l'image d'entrée
    cv::Mat m_in = cv::imread("../images/in.jpg", cv::IMREAD_UNCHANGED);
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    auto start = std::chrono::high_resolution_clock::now();


    // Allocation et copie des données sur le GPU
    unsigned char* rgb_d = nullptr;
    unsigned char* g_d = nullptr;
    unsigned char* s_d = nullptr;
    cudaMalloc(&rgb_d, 3 * rows * cols);
    cudaMalloc(&g_d, rows * cols);
    cudaMalloc(&s_d, rows * cols);
    cudaMemcpy(rgb_d, m_in.data, 3 * rows * cols, cudaMemcpyHostToDevice);

    // Définition des paramètres de grille et de bloc pour les kernels
    dim3 block(64, 8);
    dim3 grid0((cols - 1) / block.x + 1, (rows - 1) / block.y + 1);
    dim3 grid1((cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1);

    // Création des streams CUDA
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    std::size_t const sizeb = (cols*rows) * sizeof( int );

    // Appel du premier kernel
    //grayscale_boxblur_shared<<<grid1, block, block.x * block.y * sizeof(unsigned char), stream[0]>>>(rgb_d, s_d, cols, rows/2);

    // Appel du deuxième kernel
    grayscale_boxblur_shared<<<grid1, block, block.x * block.y * sizeof(unsigned char), stream[1]>>>(rgb_d+(rows*cols)/2, g_d+sizeb/2, cols, rows/2);

    // Copie du résultat final sur le CPU
    unsigned char* out = nullptr;
    cudaMallocHost(&out, rows * cols);
    //cudaMemcpyAsync(out, s_d, (rows * cols)/2, cudaMemcpyDeviceToHost, stream[0]);
    cudaMemcpyAsync(out, s_d, (rows * cols)/2, cudaMemcpyDeviceToHost, stream[0]);
    cudaMemcpyAsync(out+(rows * cols)/2, g_d, (rows * cols)/2, cudaMemcpyDeviceToHost, stream[1]);
    //cudaMemcpyAsync(out+(rows * cols)/2, g_d, (rows * cols)/2, cudaMemcpyDeviceToHost, stream[1]);
    //cudaMemcpyAsync(out+(rows * cols)/2, s_d+(rows * cols)/2, (rows * cols)/2, cudaMemcpyDeviceToHost, stream[1]);

    cv::Mat m_out( rows, cols, CV_8UC1, out );

    // Affichage du temps d'exécution
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    cv::imwrite( "out.jpg", m_out );
    // Libération de la mémoire
    cudaFree(rgb_d);
    cudaFree(g_d);
    cudaFree(s_d);
    cudaFreeHost(out);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    return 0;
}