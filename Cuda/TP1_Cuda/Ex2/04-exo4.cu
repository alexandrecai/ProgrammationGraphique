/**
 * Fill a vector of 100 ints on the GPU with consecutive values.
 */
#include <iostream>
#include <vector>


__global__ void fill( int * v, std::size_t size )
{
  // Get the id of the thread ( 0 -> 99 ).
  auto tid = threadIdx.x;
  // Each thread fills a single element of the array.
  
  v[ tid ] = tid;
}

__global__ void sum( int * v, int * v2, std::size_t size )
{
  auto tid = threadIdx.x;
  auto valv = v[ tid ];
  v[ tid ] = valv + v2[ tid ];
}


int main()
{
  std::vector< int > v( 10 );

  int * v_d = nullptr;
  int * v_d2 = nullptr;
  
  // Allocate an array an the device.
  cudaMalloc( &v_d, v.size() * sizeof( int ) );
  cudaMalloc( &v_d2, v.size() * sizeof( int ) );

  // Launch one block of 100 threads on the device.
  // In this block, threads are numbered from 0 to 99.
  fill<<< 1, 10 >>>( v_d, v.size() );
  fill<<< 1, 10 >>>( v_d2, v.size() );
  
  sum<<< 1, 10 >>>(  v_d, v_d2, v.size() );

  // Copy data from the device memory to the host memory.
  cudaMemcpy( v.data(), v_d, v.size() * sizeof( int ), cudaMemcpyDeviceToHost );

  for( auto x: v )
  {
    std::cout << x << std::endl;
  }

  cudaFree( v_d );
  cudaFree( v_d2);  

  return 0;
}
