// -*-c++-*-
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <fcntl.h>
#include <sysexits.h>
#include <unistd.h>
#include <time.h>
#include "main.h"

#include <builtin_types.h>
#include <math_constants.h>

#include "trackfit_kernel.cu"


long g_szDataIn;
long g_szDataOut;
long g_szFlag;
long g_szGood;
unsigned int *d_data_in;
unsigned int *d_data_out;
char *h_flag;
char *d_flag;

unsigned int *h_ngood;
unsigned int *d_ngood;

#define CUDA_TIMING
#define MAPPED_MEM

extern "C"
{
    void GPU_Init(unsigned long *ram, unsigned int N_WORDS_IN, unsigned int N_OUT_ARRAY_SIZE,
		  unsigned *data_in, unsigned *data_out)
    {
        //int i;
        srand(time(NULL));
	
#ifdef DEBUG_RND
        srand(1);
#endif

        *ram = 0;

        g_szDataIn  = N_WORDS_IN  * sizeof(unsigned);
        g_szDataOut = N_OUT_ARRAY_SIZE * sizeof(unsigned);
        //g_szDataOut = N_WORDS_OUT * 4 * N_BLOCKS * sizeof(unsigned);
        //g_szDataOut = N_WORDS_OUT * sizeof(unsigned);
        g_szFlag    = N_WORDS_IN  * sizeof(char);
	g_szGood    = sizeof(unsigned);

        cudaMallocHost(&h_flag, g_szFlag);
	cudaMallocHost(&h_ngood, g_szGood);

#ifdef MAPPED_MEM
	cudaHostGetDevicePointer(&d_data_in, data_in, 0);
	cudaHostGetDevicePointer(&d_data_out, data_out, 0);
#else
        cudaMalloc((void **)&d_data_in,  g_szDataIn);
        cudaMalloc((void **)&d_data_out, g_szDataOut);
#endif
	
        cudaMalloc((void **)&d_flag,     g_szFlag);
	cudaMalloc((void **)&d_ngood, g_szGood);

        *ram += g_szDataIn + g_szDataOut + g_szFlag + g_szGood;

    }

    void GPU_Destroy()
    {
        cudaFreeHost(h_flag);
        cudaFreeHost(h_ngood);
	
        cudaFree(d_data_in);
        cudaFree(d_data_out);
        cudaFree(d_flag);
	cudaFree(d_ngood);
    }


    void GPU_Trigger_CopyOnly(unsigned *data_in, unsigned *data_out, unsigned int N_BLOCKS, unsigned int N_THREADS_PER_BLOCK)
    {
        //printf("First word in (GPU): %x\n",data_in[0]);

#ifdef CUDA_TIMING
	cudaEvent_t c_start, c_stop;
	cudaEventCreate(&c_start);
	cudaEventCreate(&c_stop);
	cudaEventRecord(c_start, 0);
#endif

        //Copy to the Device
#ifndef MAPPED_MEM
	cudaMemcpy(d_data_in, data_in, g_szDataIn, cudaMemcpyHostToDevice);
#endif


        //Run the Kernel
        //kTrigger_big<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_flag, d_data_out, d_data_in);
        //kTrigger<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_flag, d_data_out, d_data_in);
        kTrigger_CopyOnly<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_flag, d_data_out, d_data_in);

        //cudaDeviceSynchronize();

        //Copy Output to Host
#ifndef MAPPED_MEM
        cudaMemcpy(data_out, d_data_out, g_szDataOut, cudaMemcpyDeviceToHost);
#else
        cudaDeviceSynchronize(); // need to synchronize CPU with GPU execution
#endif

#ifdef CUDA_TIMING
	cudaEventRecord(c_stop, 0);
	cudaEventSynchronize(c_stop);
#endif

        //printf("First word out (GPU): %x\n",data_out[1]);

#ifdef CUDA_TIMING
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
	elapsedTime *= 1000.0; // ms to us
	FILE *outFile = fopen("output/cudaTimerKernel.txt", "a");
	//FILE *outFile = fopen("output/copyHostToDevice.txt", "a");
	//FILE *outFile = fopen("output/copyKernel.txt", "a");
	//FILE *outFile = fopen("output/copyDeviceToHost.txt", "a");
	fprintf(outFile, "%f\n", elapsedTime);
	fclose(outFile);
#endif

    }

    void GPU_Trigger_big(unsigned *data_in, unsigned *data_out, unsigned int N_BLOCKS, unsigned int N_THREADS_PER_BLOCK)
    {
	//printf("First word in (GPU): %x\n",data_in[0]);

	//Copy to the Device
	cudaMemcpy(d_data_in, data_in, g_szDataIn, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_data_in, data_in, 0, cudaMemcpyHostToDevice);

	//cudaDeviceSynchronize();

	//Run the Kernel
	kTrigger_big<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_flag, d_data_out, d_data_in);
	//kTrigger<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_flag, d_data_out, d_data_in);
	//kTrigger_CopyOnly<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_flag, d_data_out, d_data_in);

	//cudaDeviceSynchronize();

	//Copy Output to Host
	cudaMemcpy(data_out, d_data_out, g_szDataOut, cudaMemcpyDeviceToHost);

	//cudaDeviceSynchronize();

	//printf("First word out (GPU): %x\n",data_out[1]);

    }

    void GPU_Trigger_SP(unsigned *n_outptr, unsigned *data_in, unsigned *data_out, unsigned int N_BLOCKS, unsigned int N_THREADS_PER_BLOCK)
    {
	//printf("First word in (GPU): %x\n",data_in[0]);
	cudaMemset(d_ngood, 0, g_szGood);
	
	//Copy to the Device
	cudaMemcpy(d_data_in, data_in, g_szDataIn, cudaMemcpyHostToDevice);

	//Run the Kernel
	kTrigger_SP<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_ngood, d_data_out, d_data_in);

	//Copy Output to Host
	cudaMemcpy(h_ngood, d_ngood, g_szGood, cudaMemcpyDeviceToHost);
	
	//printf("h_ngood = %u\n", h_ngood[0]);
	//h_ngood[0] = 10;
	*n_outptr = h_ngood[0] * 4;
	g_szDataOut = *n_outptr * sizeof(unsigned);
	//printf("data out size: %lu\n", g_szDataOut);
	cudaMemcpy(data_out, d_data_out, g_szDataOut, cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();

	//for (int i=0; i<*n_outptr; i++)
	//  printf("GPU[%u] = %x\n", i, data_out[i]);
    }


    
} // extern "C"


