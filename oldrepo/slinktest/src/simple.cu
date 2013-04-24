// -*-c-*-

#include <stdio.h>
#include "cuda_runtime.h"
#include <cutil.h>
#include <cutil_math.h>
#include "main.h"


extern "C" {
  void GPU_Init(unsigned long *ram, unsigned int **h_data_in, 
		unsigned int **h_data_out);
  void GPU_Destroy();
  void GPU_Trigger(unsigned int *data_in, unsigned int *data_out); 
}




int main()
{
  unsigned long ram;

  unsigned int *h_data_in = 0;
  unsigned int *h_data_out = 0;


  // Setup the GPU
  // select the best device
  int num_devices, device;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices));
  if ( num_devices > 1 ) {
    int max_multiprocessors = 0, max_device = 0;
    cudaDeviceProp best_prop;
    for ( device = 0; device < num_devices; ++device ) {
      cudaDeviceProp properties;
      CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
      if ( max_multiprocessors <= properties.multiProcessorCount ) {
	max_multiprocessors = properties.multiProcessorCount;
	max_device = device;
	best_prop = properties;
      }
    }
    cudaSetDevice(max_device);
    printf("Running on device %d (name %s)\n", max_device, best_prop.name);
  }
  printf("=========== init GPU ============\n");
  GPU_Init(&ram, &h_data_in, &h_data_out);
  printf("Size = %ld\n", ram);
  printf("pointers: %p %p\n", h_data_in, h_data_out);

  srand(1231217);
  for (int i = 0; i < N_RECORDS_OUT; ++i ) {
    h_data_out[i] = rand();
  }


  printf("First word in: 0x%08lx\n",h_data_out[0]);
  //=======================================
  //Run GPU/CPU Code and copy data back to host memory
  //=======================================
  GPU_Trigger(h_data_in,h_data_out);

  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUT_CHECK_ERROR("kernel invocation");

  GPU_Destroy();
  printf("=========== end ===========\n");

  return 0;
}
