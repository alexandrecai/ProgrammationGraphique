#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

/**
 * Kernel pour transformer l'image RGB en niveaux de gris.
 */
__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }
}

/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris.
 */
__global__ void laplacian_gaussian( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows )
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 1 && i < cols && j > 1 && j < rows )
  {
      /*
    auto h =     g[ (j-1)*cols + i - 1 ] -     g[ (j-1)*cols + i + 1 ]
           + 2 * g[ (j  )*cols + i - 1 ] - 2 * g[ (j  )*cols + i + 1 ]
           +     g[ (j+1)*cols + i - 1 ] -     g[ (j+1)*cols + i + 1 ];

    auto v =     g[ (j-1)*cols + i - 1 ] -     g[ (j+1)*cols + i - 1 ]
           + 2 * g[ (j-1)*cols + i     ] - 2 * g[ (j+1)*cols + i     ]
           +     g[ (j-1)*cols + i + 1 ] -     g[ (j+1)*cols + i + 1 ];

    auto res = h*h + v*v;
    res = res > 65535 ? res = 65535 : res;

    s[ j * cols + i ] = sqrtf( res );
    */


      auto res =       g[((j - 2) * cols + i - 2) ] * 0 + g[((j - 2) * cols + i -1) ] * 0 +  g[((j - 2) * cols + i) ]* -1 + g[((j - 2) * cols + i +1 ) ] * 0 + g[((j - 2) * cols + i + 2) ] *0
                  +  g[((j - 1) * cols + i - 2) ] * 0 + g[((j - 1) * cols + i -1) ] * -1 +  g[((j - 1) * cols + i) ]* -2 + g[((j - 1) * cols + i +1 ) ] * -1 + g[((j - 1) * cols + i + 2) ] *0
                  +     g[((j) * cols + i - 2) ] * -1 + g[((j) * cols + i -1) ] * -2 +  g[((j) * cols + i) ]* 16 + g[((j) * cols + i +1 ) ] * -2 + g[((j) * cols + i + 2) ] * -1
                  +    g[((j + 1) * cols + i - 2) ] * 0 + g[((j +1) * cols + i -1) ] * -1 +  g[((j + 1) * cols + i) ]* -2 + g[((j + 1) * cols + i +1 ) ] * -1 + g[((j + 1) * cols + i + 2) ] *0
                  +    g[((j + 2) * cols + i - 2) ] * 0 + g[((j + 2) * cols + i -1) ] * 0 +  g[((j + 2) * cols + i) ]* -1 + g[((j + 2) * cols + i +1 ) ] * 0 + g[((j + 2) * cols + i + 2) ] *0;

      res = res > 255 ? 255 : res;
      res = res < 0 ? 0 : res;

      s[j * cols + i] = res;
  }
}


/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris, en utilisant la mémoire shared
 * pour limiter les accès à la mémoire globale.
 */
__global__ void laplacian_gaussian_shared( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows )
{
  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  extern __shared__ unsigned char sh[];

  if( i < cols && j < rows )
  {
    sh[ lj * w + li ] = g[ j * cols + i ];
  }

  __syncthreads();

  if( i < cols -2 && j < rows-2 && li > 0 && li < (w-2) && lj > 0 && lj < (h-2) )
  {
      /*
    auto h =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj-1)*w + li + 1 ]
           + 2 * sh[ (lj  )*w + li - 1 ] - 2 * sh[ (lj  )*w + li + 1 ]
           +     sh[ (lj+1)*w + li - 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto v =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj+1)*w + li - 1 ]
           + 2 * sh[ (lj-1)*w + li     ] - 2 * sh[ (lj+1)*w + li     ]
           +     sh[ (lj-1)*w + li + 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto res = h*h + v*v;
    res = res > 65535 ? res = 65535 : res;

    s[ j * cols + i ] = sqrtf( res );
       */

      auto res =       sh[((lj - 2) * w + li - 2) ] * 0 + sh[((lj - 2) * w + li -1) ] * 0 +  sh[((lj - 2) * w + li) ]* -1 + sh[((lj - 2) * w + li +1 ) ] * 0 + sh[((lj - 2) * w + li + 2) ] *0
                       +  sh[((lj - 1) * w + li - 2) ] * 0 + sh[((lj - 1) * w + li -1) ] * -1 +  sh[((lj - 1) * w + li) ]* -2 + sh[((lj - 1) * w + li +1 ) ] * -1 + sh[((lj - 1) * w + li + 2) ] *0
                       +     sh[((lj) * w + li - 2) ] * -1 + sh[((lj) * w + li -1) ] * -2 +  sh[((lj) * w + li) ]* 16 + sh[((lj) * w + li +1 ) ] * -2 + sh[((lj) * w + li + 2) ] * -1
                       +    sh[((lj + 1) * w + li - 2) ] * 0 + sh[((lj +1) * w + li -1) ] * -1 +  sh[((lj + 1) * w + li) ]* -2 + sh[((lj + 1) * w + li +1 ) ] * -1 + sh[((lj + 1) * w + li + 2) ] *0
                       +    sh[((lj + 2) * w + li - 2) ] * 0 + sh[((lj + 2) * w + li -1) ] * 0 +  sh[((lj + 2) * w + li) ]* -1 + sh[((lj + 2) * w + li +1 ) ] * 0 + sh[((lj + 2) * w + li + 2) ] *0;

      res = res > 255 ? 255 : res;
      res = res < 0 ? 0 : res;

      s[j * cols + i] = res;


  }
}


/**
 * Kernel fusionnant le passage en niveaux de gris et la détection de contours.
 */
__global__ void grayscale_laplacian_gaussian_shared( unsigned char * rgb, unsigned char * s, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

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
 
  if( i < cols -1 && j < rows-1 && li > 0 && li < (w-1) && lj > 0 && lj < (h-1) )
  {
      auto res =       sh[((lj - 2) * w + li - 2) ] * 0 + sh[((lj - 2) * w + li -1) ] * 0 +  sh[((lj - 2) * w + li) ]* -1 + sh[((lj - 2) * w + li +1 ) ] * 0 + sh[((lj - 2) * w + li + 2) ] *0
                       +  sh[((lj - 1) * w + li - 2) ] * 0 + sh[((lj - 1) * w + li -1) ] * -1 +  sh[((lj - 1) * w + li) ]* -2 + sh[((lj - 1) * w + li +1 ) ] * -1 + sh[((lj - 1) * w + li + 2) ] *0
                       +     sh[((lj) * w + li - 2) ] * -1 + sh[((lj) * w + li -1) ] * -2 +  sh[((lj) * w + li) ]* 16 + sh[((lj) * w + li +1 ) ] * -2 + sh[((lj) * w + li + 2) ] * -1
                       +    sh[((lj + 1) * w + li - 2) ] * 0 + sh[((lj +1) * w + li -1) ] * -1 +  sh[((lj + 1) * w + li) ]* -2 + sh[((lj + 1) * w + li +1 ) ] * -1 + sh[((lj + 1) * w + li + 2) ] *0
                       +    sh[((lj + 2) * w + li - 2) ] * 0 + sh[((lj + 2) * w + li -1) ] * 0 +  sh[((lj + 2) * w + li) ]* -1 + sh[((lj + 2) * w + li +1 ) ] * 0 + sh[((lj + 2) * w + li + 2) ] *0;

      res = res > 255 ? 255 : res;
      res = res < 0 ? 0 : res;

      s[j * cols + i] = res;
  }
}


int main()
{
  cv::Mat m_in = cv::imread("../images/in.jpg", cv::IMREAD_UNCHANGED );

  //auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  //std::vector< unsigned char > g( rows * cols );
  // Allocation de l'image de sortie en RAM côté CPU.
  unsigned char * g = nullptr;
  cudaMallocHost( &g, rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g );

  // Copie de l'image en entrée dans une mémoire dite "pinned" de manière à accélérer les transferts.
  // OpenCV alloue la mémoire en interne lors de la décompression de l'image donc soit sans doute avec
  // un malloc standard.
  unsigned char * rgb = nullptr;
  cudaMallocHost( &rgb, 3 * rows * cols );
  
  std::memcpy( rgb, m_in.data, 3 * rows * cols );

  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * s_d;

  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &s_d, rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  dim3 block( 32, 4 );
  dim3 grid0( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );
  /**
   * Pour la version shared il faut faire superposer les blocs de 2 pixels
   * pour ne pas avoir de bandes non calculées autour des blocs
   * on crée donc plus de blocs.
   */
  dim3 grid1( ( cols - 1) / (block.x-2) + 1 , ( rows - 1 ) / (block.y-2) + 1 );
    
  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel uniquement.
  cudaEventRecord( start );

    /*
  // Version en 2 étapes.
  grayscale<<< grid0, block >>>( rgb_d, g_d, cols, rows );
  laplacian_gaussian<<< grid0, block >>>( g_d, s_d, cols, rows );
    */


  // Version en 2 étapes, Sobel avec mémoire shared.
  grayscale<<< grid0, block >>>( rgb_d, g_d, cols, rows );
  laplacian_gaussian_shared<<< grid1, block, block.x * block.y >>>( g_d, s_d, cols, rows );


  // Version fusionnée.
  //grayscale_laplacian_gaussian_shared<<< grid1, block, block.x * block.y >>>( rgb_d, s_d, cols, rows );

  cudaEventRecord( stop );
  
  cudaMemcpy( g, s_d, rows * cols, cudaMemcpyDeviceToHost );

  cudaEventSynchronize( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "out.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaFree( s_d);

  cudaFreeHost( g );
  cudaFreeHost( rgb );
  
  return 0;
}
