//============================================================================
// Name        : main.cpp
// Author      : Ryan Rivera and Wes Ketchum
// Version     :
// Copyright   : Created 2013 - rrivera at fnal dot gov
// Description : OpenCL version of CUDA slinktest
//============================================================================

#include <utility>
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include <CL/cl.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

#include "helperFuncs.h"
#include "clHelperFuncs.h"
#include "histo.h"

//GPU study, shared code with CUDA slinktest project
//#include "../../../slinktest/include/NodeUtils.hh"
#include "../../../slinktest/src/NodeUtils_basic.hh"
#include "../../../slinktest/src/semaphore.c"
#include "../../../slink_pulsar/s32pci64-filar/s32pci64-filar/s32pci64-filar.h"
#include "../../../slink_pulsar/s32pci64-solar/s32pci64-solar/s32pci64-solar.h"

#define FILAR_NUMBER_0 0
#define FILAR_CHANNEL_MASK 0x1
#define FILAR_CHANNEL_0 0x1
//end shared code with CUDA slinktest project

#define USE_LOCAL_GEN_DATA 1
#define DEBUG_KERNEL       0


 
typedef unsigned int	   DATAWORD; //e.g. UINT32 or float //NOTE: this must absolutely match the typedef in kernel!!!!


//============================
//
// Code conventions:
//   - functions implemented in other project .cpp files are
//     called with the namespace indicating file location.
//
//============================

using namespace std;

int main(int argc, char *argv[])
{
  ////////////////////////////////////////////
  // 
  // main() outline
  //
  // - Init
  //   - get command line args
  //   - set application cpu affinity and priority
  //   - get OpenCL platforms and their devices in a vector
  //   - lock semaphore control
  //   - allocate host memory
  //
  // - Operations
  //   - (optional loop iterating on [platforms,devices] combinations)
  //      - (optional loop(s) for varying any parameters)
  //        - (optional loop for timing data samples)
  //          - acquire data
  //          - run code on OpenCL [platform,device] queue
  //          - acquire timing statistics
  //
  // - Cleanup
  //   - unlock semaphore control
  //   - output results
  //
  //------------------------------------------
  //==========================================
  //------------------------------------------
  ////////////////////////////////////////////

  
        //running parameters   
        unsigned int N_LOOPS, N_WORDS_IN, N_WORDS_OUT, ITERATION; 
	unsigned int N_THREADS, N_BLOCKS, N_THREADS_PER_BLOCK, N_OUT_ARRAY_SIZE;
	string METHOD;

	cout << "Usage: main N_LOOPS N_WORDS N_THREADS {COPY|ALG|WEAVE|CPU|SP} ITERATION\n"
	     << endl;
        if (argc != 6) {
	    return EXIT_FAILURE;
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
	N_BLOCKS = N_THREADS/N_WORDS_IN;
	N_THREADS_PER_BLOCK = N_WORDS_IN;//N_THREADS / N_BLOCKS;
	N_OUT_ARRAY_SIZE = N_THREADS * 4;

	string kernelPath = "kernels/";
	string kernelFile = "slink.cl"; //"lesson1_kernels.cl";
	string kernelFunc;
	if(METHOD == "COPY") kernelFunc = "kTrigger_CopyOnly";
	else if(METHOD == "ALG") kernelFunc = "kTrigger_big";
	else if(METHOD == "WEAVE") kernelFunc = "kTrigger_CopyOnly";
	else if(METHOD == "CPU") kernelFunc = "kTrigger_big";
	else if(METHOD == "SP") kernelFunc = "kTrigger_CopyOnly";
	else kernelFunc = "hello";


	HELPERFUNCS::TemplateDataType<DATAWORD> dataType;

	printf("=========== parameters ========\n");
	printf("%-20s: %8s\n", "METHOD", METHOD.c_str());
	printf("%-20s: %8s\n", "KERNEL_FILE", kernelFile.c_str());
	printf("%-20s: %8s\n", "KERNEL_FUNC", kernelFunc.c_str());
	printf("%-20s: %8u\n", "N_LOOPS", N_LOOPS);
	printf("%-20s: %8s\n", "DATA_TYPE", dataType.getType().c_str());
	printf("%-20s: %8u\n", "DATA_SIZE", dataType.getSize());
	printf("%-20s: %8u\n", "N_WORDS_IN", N_WORDS_IN);
	printf("%-20s: %8u\n", "N_WORDS_OUT", N_WORDS_OUT);
	printf("%-20s: %8u\n", "N_THREADS", N_THREADS);
	printf("%-20s: %8u\n", "N_BLOCKS", N_BLOCKS);
	printf("%-20s: %8u\n", "N_THREADS_PER_BLOCK", N_THREADS_PER_BLOCK);
	printf("%-20s: %8u\n", "N_OUT_ARRAY_SIZE", N_OUT_ARRAY_SIZE);
	printf("%-20s: %8u\n", "ITERATION", ITERATION);
	//#ifdef PINNED_MEM
	//printf("%-20s: %8s\n", "MEMORY", "PINNED");
	//#endif
	printf("===============================\n");
	printf("\n\n");

        //set application cpu affinity and priority
        cout << __FANCY__ << "Setting up application cpu affinity and priority... " << endl;
	int AppEnableMask = APP_STATUS_SYS_PRIORITY|APP_STATUS_CPU_AFFINITY|APP_STATUS_SCHED|APP_STATUS_THREAD_SCHED;
	int AppSystemPriority = -20;
	int AppCpuAffinity = 0;
	int AppSchedPolicy = SCHED_FIFO;
	int AppSchedPriority = 99;
	int AppThreadSchedPolicy = SCHED_FIFO;
	int AppThreadSchedPriority = 99;

	HELPERFUNCS::initMainAppStatus(AppEnableMask,
			AppSystemPriority,
			AppCpuAffinity,
			AppSchedPolicy,
			AppSchedPriority,
			AppThreadSchedPolicy,
			AppThreadSchedPriority
			);

	//OpenCL steps:
	// - get platforms and devices
	// - get context for platform
	// - build program source
	// - set kernel entry point
	// - define input/output parameters
	// - enqueue tasks

	//get OpenCL platforms and their devices in a vector
        cout << __FANCY__ << "getting OpenCL cable platforms and their devices in a vector..." << endl;

	cl_int err;
	cl::vector< cl::Platform > platformList;
	cl::vector< cl::vector< cl::Device > *> deviceList;
	unsigned int GPU_device_index, GPU_platform_index;
	unsigned int start, finish;
	cl::Event event;

	CL_HELPERFUNCS::getPlatformsAndDevices(&platformList,&deviceList,GPU_device_index,GPU_platform_index);
	///// FOR DEBUG of OpenCL kernel code,
	///// nvidia platform does not report errors, only AMD (?)
	
	if(DEBUG_KERNEL)
	  GPU_device_index = GPU_platform_index = 0; //use AMD platform to debug kernel!
	if(METHOD == "CPU")
	  GPU_device_index = GPU_platform_index = 0; //use AMD CPU!
	
	//============================================================================
	//============================================================================
	//============================================================================
	// lock control so no one else can run at the same time and crash the machine
	key_t key = (key_t) 0xdeadface;
	int semid;
	
	if ((semid = initsem(key, 1)) == -1) {
	  perror("initsem");
	  exit(1);
	}
	cout << __FANCY__ << "Attempting to lock control..."; lock(semid);
	//============================================================================
	//============================================================================
	//============================================================================
	
	//allocate host memory
	char *hostMemIn = new char[N_WORDS_IN*dataType.getSize()];
	char *hostMemOut = new char[N_OUT_ARRAY_SIZE*dataType.getSize()];
	
	//for solar/filar
	unsigned *dataptr_solar;
	int filar_err, fsize;
	int reqfifobufcount = 0;
	int reqfifobufused = 1;

	if(!USE_LOCAL_GEN_DATA) //setup solar and filar
	{
   	    solar_setup(0);
	    dataptr_solar = solar_getbuffer(0);
	    
	    filar_setup_hola(FILAR_NUMBER_0, FILAR_CHANNEL_MASK);

	    // ARM REQ FIFOs before going to event loop
	    filar_setreq(FILAR_NUMBER_0, FILAR_CHANNEL_0, 4);

	}


	//(optional loop iterating on [platforms,devices] combinations)
	for(unsigned int plat_i = GPU_platform_index; plat_i == GPU_platform_index; ++plat_i) 
	{	  
	     cl::Platform platform = platformList[plat_i]; //get current platform to simplify calls

	     //====================================
	     //get context for platform
	     cl_context_properties cprops[3] = {
	       CL_CONTEXT_PLATFORM, 
	       (cl_context_properties)(platform()),
	       0};

	     cl::Context context(
			      CL_DEVICE_TYPE_ALL,
			      cprops,
			      NULL,
			      NULL,
			      &err);

	     CL_HELPERFUNCS::checkErr(err, "Context::Context()");
	  
	     for(unsigned int dev_i = GPU_device_index; dev_i == GPU_device_index; ++dev_i) 
	     {
	         //run specified kernel on device based on plat_i and dev_i
	         cout << __FANCY__ << "selected [Platform, Device] is [" << plat_i << "," << dev_i << "]" << 
		   " type=" << (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)?"GPU":"CPU") << endl;

		 cl::Device device = (*deviceList[plat_i])[dev_i]; //get current device to simplify calls

		 cl::CommandQueue queue(context, device, 0, &err);
		 CL_HELPERFUNCS::checkErr(err, "CommandQueue::CommandQueue()");

		 //====================================
		 //build program source
		 std::ifstream file((kernelPath + kernelFile).c_str());
		 CL_HELPERFUNCS::checkErr(file.is_open() ? CL_SUCCESS:-1, (kernelPath + kernelFile).c_str());
		 

		 //TODO-RAR add build options
		 string prog(std::istreambuf_iterator<char>(file),
				  (std::istreambuf_iterator<char>()));
		 string buildOptions;
		 { // create preprocessor defines for the kernel
		   char buf[256]; 
		   sprintf(buf,"-D DATAWORD=%s ", dataType.getType().c_str());
		   buildOptions = buf;
		   cout << __FANCY__ << "buildOptions = " << buildOptions << endl;
		 }

		 cl::Program::Sources source(1,std::make_pair(prog.c_str(), prog.length()));

		 cl::Program program(context, source);

		 err = program.build(*(deviceList[plat_i]),buildOptions.c_str());
	      
		 CL_HELPERFUNCS::displayBuildLog(program, device);
		 CL_HELPERFUNCS::checkErr(err, "Build::Build()");

		 //====================================
		 //set kernel entry point
		 cl::Kernel kernel(program, kernelFunc.c_str(), &err);
		 CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
		 cout << __FANCY__ << "Entry point set to " << kernelFunc << endl;
		 
		 // Get the maximum work group size for executing the kernel on the device

		 cl::NDRange maxWorkGroupSize;
		 err = kernel.getWorkGroupInfo(device(),
					       CL_KERNEL_WORK_GROUP_SIZE,
					       &maxWorkGroupSize); //returns 3 dimensional size
 
		 cout << __FANCY__ << "Max work group size = [" << 
		   maxWorkGroupSize[0] << "," << 
		   maxWorkGroupSize[1] << "," <<
		   maxWorkGroupSize[2] << "]" << endl;
		 if(N_THREADS_PER_BLOCK > maxWorkGroupSize[0]) N_THREADS_PER_BLOCK = maxWorkGroupSize[0];
		 cout << __FANCY__ << "Work group size(N_THREADS_PER_BLOCK) = " << N_THREADS_PER_BLOCK << endl;

		 //(optional loop for timing data samples)
		 for(unsigned int loop = 0; loop < N_LOOPS; ++loop) 
		 {
		     //====================================
		     //define input/output parameters
		     //in memory


		     //acquire data
		     if(USE_LOCAL_GEN_DATA)
		     {
		         cout << __FANCY__ << "Local random data generation of " 
			      << N_WORDS_IN << " words" << endl;

			 for(unsigned int i=0;i<N_WORDS_IN*dataType.getSize();++i)
			     hostMemIn[i] = i%26+65;//alpha chars			   
		     }
		     else //get pulsar data
		     {
			 cout << __FANCY__ << "Acquiring pulsar data of " 
			      << N_WORDS_IN << " words" << endl;

			 // Write REQ FIFO for Cluster channel
			 if (reqfifobufcount == reqfifobufused - 1)
			     filar_setreq(FILAR_NUMBER_0, FILAR_CHANNEL_0, reqfifobufused);

			 // wait for data on the filar
			 while (!fsize && !filar_err)
			     fsize = filar_receive_channel_ptr(FILAR_NUMBER_0, &filar_err,
							       FILAR_CHANNEL_0, 
							       (unsigned int *)hostMemIn);

			 if (!fsize && filar_err) 
			 {
			     printf("Received Error: %d", filar_err);
			     break;
			 }
		     }

		     cl::Buffer inCL(
				     context,
				     CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
				         0:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				     N_WORDS_IN*dataType.getSize(),
				     hostMemIn,
				     &err);
		     CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() IN");

		     cl::Buffer outCL(
				      context,
				      CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
				      0:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
				      N_OUT_ARRAY_SIZE*dataType.getSize(),
				      hostMemOut,
				      &err);
		     CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT");


		     err = kernel.setArg(0, inCL);
		     err = kernel.setArg(1, outCL);
		     CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");

		     //====================================
		     //enqueue tasks

		     rdtscl(start);
		     err = queue.enqueueWriteBuffer(
						    inCL,
						    CL_TRUE,
						    0,
						    N_WORDS_IN*dataType.getSize(),
						    hostMemIn);
		     CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueWriteBuffer()");

		     err = queue.enqueueNDRangeKernel(
						      kernel,
						      cl::NullRange,
						      cl::NDRange(N_THREADS),
						      cl::NDRange(N_THREADS_PER_BLOCK),
						      NULL,
						      &event);
		     CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");

		     event.wait();
		     err = queue.enqueueReadBuffer(
						   outCL,
						   CL_TRUE,
						   0,
						   N_OUT_ARRAY_SIZE*dataType.getSize(),
						   hostMemOut);
		     CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueReadBuffer()");

		     if(!USE_LOCAL_GEN_DATA) //for solar and filar
		     {
			 //Send result data out on the solar
   		         solar_send_ptr(N_OUT_ARRAY_SIZE*dataType.getSize(), 
				      (unsigned int *)hostMemOut, 0);
		     }

		     rdtscl(finish);
		     float time_us = tstamp_to_us(start, finish);

		     //TODO - verify correctness of result?

		     //display a few chars
		     if(100 < N_OUT_ARRAY_SIZE*dataType.getSize())
		       hostMemOut[100] = '\0'; //end string
		     else
		       hostMemOut[N_OUT_ARRAY_SIZE*dataType.getSize()-1] = '\0'; //end string
		     cout << "Result:" << endl << endl << hostMemOut << endl << endl;

		     cout << "Time: " << time_us << " us" << endl;
	
		     if(!USE_LOCAL_GEN_DATA) //for solar and filar
		     {
		         ++reqfifobufcount;
			 if(reqfifobufcount == reqfifobufused)
			   reqfifobufcount = 0;
		     }
		 }
	     }
	}

	//cleanup

	delete[] hostMemIn;
	delete[] hostMemOut;

	//TODO - necessary to release opencl C++ binding objects??

	//============================================================================
	//============================================================================
	//============================================================================
	cout << __FANCY__ << "Attempting to unlock control... "; unlock(semid);
	//============================================================================
	//============================================================================
	//============================================================================

	return EXIT_SUCCESS;
}



