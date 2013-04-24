//#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <fcntl.h>
#include <sysexits.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include <linux/types.h>
#include "cuda_runtime.h"
#include "main.h"

//include files for slink
#include "s32pci64-filar.h"
#include "s32pci64-solar.h"
#include "NodeUtils_basic.hh"

#include "semaphore.c"

//void GPU_Init(unsigned long *ram, unsigned int N_WORDS_IN, unsigned int N_OUT_ARRAY_SIZE);
void GPU_Init(unsigned long *ram, unsigned int N_WORDS_IN, unsigned int N_OUT_ARRAY_SIZE, unsigned *data_in, unsigned *data_out);
void GPU_Destroy();
void GPU_Trigger_CopyOnly(unsigned *data_in, unsigned *data_out, unsigned int N_BLOCKS, unsigned int N_THREADS_PER_BLOCK);
void GPU_Trigger_big(unsigned *data_in, unsigned *data_out, unsigned int N_BLOCKS, unsigned int N_THREADS_PER_BLOCK);
void GPU_Trigger_SP(unsigned *n_outptr, unsigned *data_in, unsigned *data_out, unsigned int N_BLOCKS, unsigned int N_THREADS_PER_BLOCK);

#include "trackfit_host.c"
//void CPU_Trigger(unsigned *data_in, unsigned *data_out);

#define FILAR_NUMBER_0 0
#define FILAR_CHANNEL_MASK 0x1
#define FILAR_CHANNEL_0 0x1

#define DEBUG_RND

//#define CUDA_TIMING
//#define PINNED_MEM
#define MAPPED_MEM


unsigned int N_LOOPS, N_WORDS_IN, N_WORDS_OUT, ITERATION;
unsigned int N_THREADS, N_BLOCKS, N_THREADS_PER_BLOCK, N_OUT_ARRAY_SIZE;
char *METHOD;

int main(int argc, char *argv[])
{
    if (argc != 6) {
        printf("Usage: main N_LOOPS N_WORDS N_THREADS {COPY|ALG|WEAVE|CPU|SP} ITERATION\n");
	return 0;
	/*
	if (argc == 2) {
	    if (strcmp(argv[1], "help") == 0)
		return 0;
	}
	printf("Using defaults\n");
	N_LOOPS = 10000;
	N_WORDS_IN = 500;
	N_THREADS = 16000;
	METHOD = "ALG";
	ITERATION = 0;
	*/
    } else {
	N_LOOPS = atoi(argv[1]);
	N_WORDS_IN = atoi(argv[2]);
	N_THREADS = atoi(argv[3]);
	METHOD = argv[4];
	ITERATION = atoi(argv[5]);
    }

    N_WORDS_OUT = N_WORDS_IN;
    //N_WORDS_OUT = 1;
    N_BLOCKS = 32;
    N_THREADS_PER_BLOCK = N_THREADS / N_BLOCKS;
    N_OUT_ARRAY_SIZE = N_THREADS * 4;
    //N_OUT_ARRAY_SIZE = 64000;

    
    // initialize necessary variables

    int output_array[N_LOOPS];
    /*
    std::ofstream fileout;
    char file_name[80];
    time_t rawtime; struct tm * timeinfo;
    time ( &rawtime ); timeinfo = localtime ( &rawtime );
    strftime(file_name,80,"output_LOGS/output_%b%d_%H%M_%S.log",timeinfo);
    puts(file_name);
    fileout.open(file_name);
    */

    // lock control so no one else can run at the same time and crash the machine
    key_t key = (key_t) 0xdeadface;
    int semid;

    if ((semid = initsem(key, 1)) == -1) {
        perror("initsem");
        exit(1);
    }
    printf("Trying to gain control...\n");
    lock(semid);

    // set scheduling priority & CPU affinity
    struct sched_param p;
    p.sched_priority = 99;
    if (sched_setscheduler(0, SCHED_FIFO, &p)) {
	perror("setscheduler");
	return -1;
    }
    if (sched_getparam(0, &p) == 0)
	printf("Running with scheduling priority = %d\n", p.sched_priority);
    
    unsigned long mask;
    if (sched_getaffinity(0, sizeof(mask), (cpu_set_t*)&mask) < 0) {
	perror("sched_getaffinity");
    }
    printf("my affinity mask is: %08lx\n", mask);
    
    mask = 1; // processor 1 only
    if (sched_setaffinity(0, sizeof(mask), (cpu_set_t*)&mask) < 0) {
	perror("sched_setaffinity");
	return -1;
    }

    if (sched_getaffinity(0, sizeof(mask), (cpu_set_t*)&mask) < 0) {
	perror("sched_getaffinity");
    }
    printf("my affinity mask is: %08lx\n", mask);

    printf("=========== parameters ========\n");
    printf("%-20s: %8s\n", "METHOD", METHOD);
    printf("%-20s: %8u\n", "N_LOOPS", N_LOOPS);
    printf("%-20s: %8u\n", "N_WORDS_IN", N_WORDS_IN);
    printf("%-20s: %8u\n", "N_WORDS_OUT", N_WORDS_OUT);
    printf("%-20s: %8u\n", "N_THREADS", N_THREADS);
    printf("%-20s: %8u\n", "N_BLOCKS", N_BLOCKS);
    printf("%-20s: %8u\n", "N_THREADS_PER_BLOCK", N_THREADS_PER_BLOCK);
    printf("%-20s: %8u\n", "N_OUT_ARRAY_SIZE", N_OUT_ARRAY_SIZE);
    printf("%-20s: %8u\n", "ITERATION", ITERATION);

#ifdef PINNED_MEM
    printf("%-20s: %8s\n", "MEMORY", "PINNED");
#endif
#ifdef MAPPED_MEM
    printf("%-20s: %8s\n", "MEMORY", "MAPPED");
#endif

    printf("===============================\n");
    printf("\n\n");

    //clock_t start, finish;
    __u32 start, finish, start_gpu, finish_gpu;
    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);


    unsigned long ram = 0;
    unsigned *dataptr_solar;

    unsigned *data;
    unsigned *data_send_GPU;

#ifdef PINNED_MEM
    //cudaHostAlloc((void**)&data, N_WORDS_IN * sizeof(unsigned), cudaHostAllocDefault);
    data = malloc(N_WORDS_IN * sizeof(unsigned));
    
    cudaHostAlloc((void**)&data_send_GPU, N_OUT_ARRAY_SIZE * sizeof(unsigned),
		  cudaHostAllocDefault);
    /*
    cudaMallocHost(&data, N_WORDS_IN * sizeof(unsigned));
    cudaMallocHost(&data_send_GPU, N_OUT_ARRAY_SIZE * sizeof(unsigned));
    */
#endif

#ifdef MAPPED_MEM

    cudaHostAlloc((void**)&data, N_WORDS_IN * sizeof(unsigned),
//		  cudaHostAllocMapped);
		  cudaHostAllocMapped | cudaHostAllocWriteCombined);
    
    cudaHostAlloc((void**)&data_send_GPU, N_OUT_ARRAY_SIZE * sizeof(unsigned),
		  cudaHostAllocMapped);
//		  cudaHostAllocMapped | cudaHostAllocWriteCombined);

    
#else
    data = malloc(N_WORDS_IN * sizeof(unsigned));
    data_send_GPU = malloc(N_OUT_ARRAY_SIZE * sizeof(unsigned));
#endif
    
    unsigned data_send_CPU[N_OUT_ARRAY_SIZE];

    unsigned adder = 0;
    int kf;

#ifdef CUDA_TIMING
    // open the output file
    char outFileStr[100];
    sprintf(outFileStr, "output/%dW_%dT_%s_%d.txt", N_WORDS_IN, N_THREADS, METHOD, ITERATION);
    
    FILE *outFile = fopen(outFileStr, "w");
#endif

    //do solar setup procedure
    solar_setup(0);
    dataptr_solar = solar_getbuffer(0);

    //Now do filar setup.
    filar_setup_hola(FILAR_NUMBER_0, FILAR_CHANNEL_MASK);
    int reqfifobufcount = 0;
    int reqfifobufused = 1;

    // ARM REQ FIFOs before going to event loop
    filar_setreq(FILAR_NUMBER_0, FILAR_CHANNEL_0, 4);
    //return 0;

    //Now let's setup up the GPU
    cudaSetDevice(0);
    printf("=========== init GPU ============\n");
    //GPU_Init(&ram, N_WORDS_IN, N_OUT_ARRAY_SIZE);
    GPU_Init(&ram, N_WORDS_IN, N_OUT_ARRAY_SIZE, data, data_send_GPU);
    //  printf("RAM on single device %f B\n\n", ram);

    printf("=========== Starting Internal Pre-Run ============\n");
    GPU_Trigger_CopyOnly(data, data_send_GPU, N_BLOCKS, N_THREADS_PER_BLOCK);
    printf("=========== OK, ready to go! ============\n");

    //=======================================
    //  Time for the event loop
    //=======================================
    int print = 0;
    int check = 0; //does comparison of GPU to CPU calculations
    int filar_err, fsize;
    int nevent = 0;

    while (nevent < N_LOOPS) {
        fsize = 0; filar_err = 0;

        if (print)
            if (nevent % 1 == 0) printf("Processed %d events.\n", nevent);

        // Write REQ FIFO for Cluster channel
        if (reqfifobufcount == reqfifobufused - 1)
            filar_setreq(FILAR_NUMBER_0, FILAR_CHANNEL_0, reqfifobufused);

        // wait for data on the filar
        while (!fsize && !filar_err)
            fsize = filar_receive_channel_ptr(FILAR_NUMBER_0, &filar_err,
                                              FILAR_CHANNEL_0, data);

        if (!fsize && filar_err) {
            printf("Received Error: %d", filar_err);
            break;
        }

        //rdtscl(start_gpu);

#ifdef CUDA_TIMING
	cudaEventRecord(c_start, 0);
#endif
	
        //=======================================
        //Run GPU/CPU Code
        //=======================================

        //printf("First word in (CPU): %x\n", data[0]);
        //printf("Going to run %d blocks and %d threads per block\n", N_BLOCKS, N_THREADS_PER_BLOCK);

	// todo: switch these to call a function pointer to avoid the strcmp for each event

	if (strcmp(METHOD, "ALG") == 0)
	    GPU_Trigger_big(data, data_send_GPU, N_BLOCKS, N_THREADS_PER_BLOCK);
	else if (strcmp(METHOD, "COPY") == 0)
	    GPU_Trigger_CopyOnly(data, data_send_GPU, N_BLOCKS, N_THREADS_PER_BLOCK);
	else if (strcmp(METHOD, "WEAVE") == 0) {
	    if (nevent % 2 == 0)
		GPU_Trigger_big(data, data_send_GPU, N_BLOCKS, N_THREADS_PER_BLOCK);
	    else
		GPU_Trigger_CopyOnly(data, data_send_GPU, N_BLOCKS, N_THREADS_PER_BLOCK);
	}
	else if (strcmp(METHOD, "CPU") == 0) {
	    CPU_Trigger_big(data, data_send_CPU, N_THREADS, N_THREADS_PER_BLOCK);
	}
	else if (strcmp(METHOD, "SP") == 0) {
	    unsigned int n;
	    GPU_Trigger_SP(&n, data, data_send_GPU, N_BLOCKS, N_THREADS_PER_BLOCK);
	    N_WORDS_OUT = n;
	    //printf("N_WORDS_OUT = %u\n", N_WORDS_OUT);
	}

#ifdef CUDA_TIMING
	cudaEventRecord(c_stop, 0);
	cudaEventSynchronize(c_stop);
#endif

	if (N_WORDS_OUT == 0) {
	    N_WORDS_OUT = 1; // need to send something
	    memset(data_send_GPU, 0, sizeof(unsigned));
	}
	
        //Send out on the solar
        solar_send_ptr(N_WORDS_OUT, data_send_GPU, 0); //anything with GPU
        //solar_send_ptr(N_WORDS_OUT, data, 0);  //for doing IO studies
        //solar_send_ptr(N_WORDS_OUT, data_send, 0);

	if (print)
	    printf("sent %d words\n", N_WORDS_OUT);
	
        rdtscl(start);

#ifdef CUDA_TIMING
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
	elapsedTime *= 1000.0; // ms to us
	fprintf(outFile, "%f\n", elapsedTime);
#endif
	
        if (check) {
	    if (strcmp(METHOD, "WEAVE") == 0) {
		if (nevent % 2 == 0)
		    CPU_Trigger_big(data, data_send_CPU, N_THREADS, N_THREADS_PER_BLOCK);
		else
		    CPU_Trigger_CopyOnly(data, data_send_CPU, N_THREADS, N_THREADS_PER_BLOCK);
	    } else {
		CPU_Trigger_CopyOnly(data, data_send_CPU, N_THREADS, N_THREADS_PER_BLOCK);
		//CPU_Trigger_big(data, data_send_CPU, N_THREADS, N_THREADS_PER_BLOCK);
	    }
        }
	
        nevent++;
        reqfifobufcount++;
        if (reqfifobufcount == reqfifobufused)
            reqfifobufcount = 0;

        for (kf = 0; kf < N_WORDS_OUT; kf++) {
            adder += data_send_GPU[kf];

            if (check) {
                if (data_send_GPU[kf] != data_send_CPU[kf]) {
                    printf("Error!!!!!! Disagreement in output word %d\n\tCPU = %x \t GPU = %x \n", kf, data_send_CPU[kf], data_send_GPU[kf]);
                    return 4;
                }
                if (print && kf % 4 == 0)
                    printf("Input = %x \t CPU = %x \t GPU = %x \n", data[kf / 4], data_send_CPU[kf], data_send_GPU[kf]);
            }
	    else if (print)
		printf("Input = %x \t GPU = %x \n", data[kf / 4], data_send_GPU[kf]);
        }

        //printf("Total Output = %ud\n",adder);
        output_array[nevent - 1] = adder;
        adder = 0;

	/*
	if (nevent == N_LOOPS) {
	    int i=0;
	    while (i < N_LOOPS) {
		fileout << "Total Output = " << hex << output_array[i];
		i++;
	    }
	}
	*/
	
        //float time_us_gpu = tstamp_to_us(start_gpu,start);
        //printf("Internal Latency = %f\tTotal output = %x (%x)\n", time_us_gpu, data_send_GPU[0], data_send_CPU[0]);

        rdtscl(finish);

        float time_us = tstamp_to_us(start, finish);
        //if (time_us > 3000) {
	if (time_us > 8000) {
            printf("ERROR: Took %f us after send...\nBetter Quit!", time_us);
            return 5;
        }

    }

    cudaEventDestroy(c_start);
    cudaEventDestroy(c_stop);

    GPU_Destroy();

#ifdef PINNED_MEM
    cudaFreeHost(data);
    cudaFreeHost(data_send_GPU);
#endif
    
    printf("=========== end ===========\n");

#ifdef CUDA_TIMING
    fclose(outFile);
#endif
    
    printf("Unlocking control...\n");
    unlock(semid);

    return 0;
}



