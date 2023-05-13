#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

__global__ void grayscale_gaussian_blur_shared( unsigned char * rgb, unsigned char * s, std::size_t cols, std::size_t rows ) {
    auto i = blockIdx.x * (blockDim.x-7) + threadIdx.x;
    auto j = blockIdx.y * (blockDim.y-7) + threadIdx.y;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    auto w = blockDim.x;
    auto h = blockDim.y;

    extern __shared__ unsigned char sh[];

    if( i < cols && j < rows ) {
        sh[ lj * w + li ] = (
                307 * rgb[ 3 * ( j * cols + i ) ]
                + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
                + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
        ) >> 10;
    }

    /**
     * Il faut synchroniser tous les warps (threads) du bloc pour être certain que le niveau de gris est calculé
     * par tous les threads du bloc avant de pouvoir accéder aux données des pixels voisins.
     */
    __syncthreads();

  if( i < cols -3 && j < rows-3 && li > 3 && li < (w-3) && lj > 3 && lj < (h-3) )
  {
    auto total =   
                      0 * sh[((lj - 3) * w + li - 3) ]  +  0 * sh[((lj - 3) * w + li - 2) ] +   0 * sh[((lj - 3) * w + li - 1) ] +   5 * sh[((lj - 3) * w + li) ] +   0 * sh[((lj - 3) * w + li + 1) ]  +  0 * sh[((lj - 3) * w + li + 2) ] + 0 * sh[((lj - 3) * w + li + 3) ]
                    + 0 * sh[((lj - 2) * w + li - 3) ]  +  5 * sh[((lj - 2) * w + li - 2) ] +  18 * sh[((lj - 2) * w + li - 1) ] +  32 * sh[((lj - 2) * w + li) ] +  18 * sh[((lj - 2) * w + li + 1) ]  +  5 * sh[((lj - 2) * w + li + 2) ] + 0 * sh[((lj - 2) * w + li + 3) ]
                    + 0 * sh[((lj - 1) * w + li - 3) ]  + 18 * sh[((lj - 1) * w + li - 2) ] +  64 * sh[((lj - 1) * w + li - 1) ] + 100 * sh[((lj - 1) * w + li) ] +  64 * sh[((lj - 1) * w + li + 1) ]  + 18 * sh[((lj - 1) * w + li + 2) ] + 0 * sh[((lj - 1) * w + li + 3) ]
                    + 5 * sh[((lj) * w + li - 3) ]      + 32 * sh[((lj) * w + li - 2) ]     + 100 * sh[((lj) * w + li - 1) ]     + 100 * sh[((lj) * w + li) ]     + 100 * sh[((lj) * w + li + 1) ]      + 32 * sh[((lj) * w + li + 2) ]     + 5 * sh[((lj) * w + li + 3) ]
                    + 0 * sh[((lj + 1) * w + li - 3) ]  + 18 * sh[((lj + 1) * w + li - 2) ] +  64 * sh[((lj + 1) * w + li - 1) ] + 100 * sh[((lj + 1) * w + li) ] +  64 * sh[((lj + 1) * w + li + 1) ]  + 18 * sh[((lj + 1) * w + li + 2) ] + 0 * sh[((lj + 1) * w + li + 3) ]
                    + 0 * sh[((lj + 2) * w + li - 3) ]  +  5 * sh[((lj + 2) * w + li - 2) ] +  18 * sh[((lj + 2) * w + li - 1) ] +  32 * sh[((lj + 2) * w + li) ] +  18 * sh[((lj + 2) * w + li + 1) ]  +  5 * sh[((lj + 2) * w + li + 2) ] + 0 * sh[((lj + 2) * w + li + 3) ]
                    + 0 * sh[((lj + 3) * w + li - 3) ]  +  0 * sh[((lj + 3) * w + li - 2) ] +   0 * sh[((lj + 3) * w + li - 1) ] +   5 * sh[((lj + 3) * w + li) ] +   0 * sh[((lj + 3) * w + li + 1) ]  +  0 * sh[((lj + 3) * w + li + 2) ] + 0 * sh[((lj + 3) * w + li + 3) ]
                    ;


    auto res = total/1068;

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
    dim3 grid1((cols - 1) / (block.x - 7) + 1, (rows - 1) / (block.y - 7) + 1);

    // Création des streams CUDA
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    // Appel du premier kernel
    grayscale_gaussian_blur_shared<<<grid1, block, block.x * (block.y+2) * sizeof(unsigned char), stream[0]>>>(rgb_d, s_d, cols, rows/2+3);

    // Appel du deuxième kernel
    grayscale_gaussian_blur_shared<<<grid1, block, block.x * (block.y+2) * sizeof(unsigned char), stream[1]>>>(rgb_d+(((rows*cols*3)/2)-cols*3*4), g_d, cols, rows/2+4);

    // Copie du résultat final sur le CPU
    unsigned char* out = nullptr;
    cudaMallocHost(&out, rows * cols);

    cudaMemcpyAsync(out, s_d, (rows * cols)/2, cudaMemcpyDeviceToHost, stream[0]);
    cudaMemcpyAsync(out+(rows * cols)/2, g_d+cols*4, (rows * cols)/2, cudaMemcpyDeviceToHost, stream[1]);


    cv::Mat m_out( rows, cols, CV_8UC1, out );

    // Affichage du temps d'exécution
    cudaDeviceSynchronize();

    //cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: 0." << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
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