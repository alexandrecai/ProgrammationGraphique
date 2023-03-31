#include <stdio.h>

int main(){
  
  int nbDevices;
  cudaGetDeviceCount(&nbDevices);

  printf("Number of devices : %d\n", nbDevices);

  for(int i=0; i<nbDevices;i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);

    printf("Device number : %d\n", i);
    printf("Device name : %d\n", prop.name);
    printf("MemoryClockRate (MHz) : %d\n", prop.memoryClockRate/1024);
    printf("Memory Bus Width (bits) : %d\n", prop.memoryBusWidth);

 
  } 
}
