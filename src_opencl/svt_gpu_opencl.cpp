#include <unistd.h>
#include <sys/time.h>
#include "semaphore.c"
#include "svt_gpu_opencl.h"

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include <CL/cl.hpp>
#include "helperFuncs.h"
#include "clHelperFuncs.h"
#include <string>
#include <fstream>
#include <iterator>

typedef unsigned int	   DATAWORD; //e.g. UINT32 or float //NOTE: this must absolutely match the typedef in kernel!!!!

/*
__kernel void init_arrays_GPU (fout_arrays* fout_dev, evt_arrays* evt_dev, int* events ) {    
}
*/

void svt_GPU(tf_arrays_t tf, unsigned int *data_in, int n_words) {

}

void set_outcable(tf_arrays_t tf) {

  svtsim_cable_copywords(tf->out, 0, 0);
  for (int ie=0; ie < tf->totEvts; ie++) {
    // insert data in the cable structure, how to "sort" them?!?
    // data should be insert only if fout_ntrks > 0
    for (int nt=0; nt < tf->fout_ntrks[ie]; nt++)
      svtsim_cable_addwords(tf->out, tf->fout_gfword[ie][nt], NTFWORDS);  
      // insert end word in the cable
    svtsim_cable_addword(tf->out, tf->fout_ee_word[ie]); 
  }
 
}

void help(char* prog) {

  printf("Use %s [-i fileIn] [-o fileOut] [-s cpu || gpu] [-h] \n\n", prog);
  printf("  -i fileIn       Input file (Default: hbout_w6_100evts).\n");
  printf("  -o fileOut      Output file (Default: gfout.txt).\n");
  printf("  -s cpu || gpu   Switch between CPU or GPU version (Default: gpu).\n");
  printf("  -h              This help.\n");

}


int main(int argc, char* argv[]) {

  int c;
  char* fileIn = "hbout_w6_100evts";
  char* fileOut = "gfout.txt";
  char* where = "gpu";

  while ( (c = getopt(argc, argv, "i:s:o:h")) != -1 ) {
    switch(c) {
      case 'i': 
        fileIn = optarg;
	      break;
      case 'o':
        fileOut = optarg;
        break;
	    case 's': 
        where = optarg;
	      break;
      case 'h':
        help(argv[0]);
        return 0;
    }
  }

  if (access(fileIn, 0) == -1) {
    printf("File %s doesn't exist.\n", fileIn);
    return 1;
  }

  struct timeval tBegin, tEnd;
  struct timeval ptBegin, ptEnd;


  // read input file
  printf("Opening file %s\n", fileIn);
  FILE* hbout = fopen(fileIn,"r");

  if(hbout == NULL) {
    printf("Cannot open input file\n");
    exit(1);
  }

  unsigned int hexaval;
  unsigned int *data_send = (unsigned int*)malloc(2500000*sizeof(unsigned));
  if ( data_send == (unsigned int*) NULL ) {
    perror("malloc");
    return 2;
  }
  
  char word[16];
  int k=0;
  while (fscanf(hbout, "%s", word) != EOF) {
    hexaval = strtol(word,NULL,16);
    data_send[k] = hexaval;
    k++;
  }

  fclose(hbout);

  //set application cpu affinity and priority
  std::cout << __FANCY__ << "Setting up application cpu affinity and priority... " << std::endl;
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

  tf_arrays_t tf;

  gf_init(&tf);
  svtsim_fconread(tf);

  struct evt_arrays *evt; int totEvts;
  struct fep_arrays *fep_dev = new fep_arrays;
  struct extra_data *edata_dev = new extra_data;
  struct fit_arrays *fit_dev = new fit_arrays;

  gf_init_evt(&evt);

  gettimeofday(&tBegin, NULL);

  if ( strcmp(where,"cpu") == 0 ) { // CPU
    printf("Start work on CPU..... \n");
    
    gettimeofday(&ptBegin, NULL);
    gf_fep_unpack(tf, k, data_send);
    gettimeofday(&ptEnd, NULL);
    printf("Time to CPU unpack: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    for(int ie=0; ie<NEVTS; ie++){
      printf("\nEvent %d, nroads = %d",ie,tf->evt_nroads[ie]);
    }

    gettimeofday(&ptBegin, NULL);
    gf_fep_comb(tf);
    gettimeofday(&ptEnd, NULL);
    printf("Time to CPU comb: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    for(int ie=0; ie<NEVTS; ie++){
      printf("\nEvent %d\n",ie);
      for(int ir=0; ir<MAXROAD; ir++){
	if(tf->fep_ncmb[ie][ir]!=0)
	  printf("\n\tRoad %d, ncomb = %d",ir,tf->fep_ncmb[ie][ir]);
      }
    }

    gettimeofday(&ptBegin, NULL);
    gf_fit(tf);
    gf_comparator(tf);
    gettimeofday(&ptEnd, NULL);
    printf("Time to CPU fit: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    printf(".... fits %d events! \n", tf->totEvts);

  } else { // GPU

    HELPERFUNCS::TemplateDataType<DATAWORD> dataType;

    //get OpenCL platforms and their devices in a vector
    std::cout << __FANCY__ << "getting OpenCL cable platforms and their devices in a vector..." << std::endl;
    
    cl_int err;
    cl::vector< cl::Platform > platformList;
    cl::vector< cl::vector< cl::Device > *> deviceList;
    unsigned int dev_i, plat_i;
    unsigned int start, finish;
    float avgTime;
    cl::Event event;
    
    CL_HELPERFUNCS::getPlatformsAndDevices(&platformList,&deviceList,dev_i,plat_i);

    //this is for GTX 285...
    dev_i = 0;
    plat_i = 1;	
    
    //this is for CPU...
    //dev_i = 0;
    //plat_i = 0;	

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
    std::cout << __FANCY__ << "Attempting to lock control..."; lock(semid);



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

    //run specified kernel on device based on plat_i and dev_i
    std::cout << __FANCY__ << "selected [Platform, Device] is [" << plat_i << "," << dev_i << "]" << 
      " type=" << (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)?"GPU":"CPU") << std::endl;
    
    cl::Device device = (*deviceList[plat_i])[dev_i]; //get current device to simplify calls
    
    cl::CommandQueue queue(context, device, 0, &err);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::CommandQueue()");


    //====================================
    //build program source
    std::string kernel_file = "gf_fep.cl";
    std::ifstream file((kernel_file).c_str());
    CL_HELPERFUNCS::checkErr(file.is_open() ? CL_SUCCESS:-1, (kernel_file).c_str());
    
    
    //TODO-RAR add build options
    std::string prog(std::istreambuf_iterator<char>(file),
		     (std::istreambuf_iterator<char>()));
    std::string buildOptions;
    { // create preprocessor defines for the kernel
      char buf[256]; 
      sprintf(buf,"-cl-mad-enable -D DATAWORD=%s ", dataType.getType().c_str());
      buildOptions = buf;
      std::cout << __FANCY__ << "buildOptions = " << buildOptions << std::endl;
    }
    
    cl::Program::Sources source(1,std::make_pair(prog.c_str(), prog.length()));
    
    cl::Program program(context, source);
    
    err = program.build(*(deviceList[plat_i]),buildOptions.c_str());
	      
    CL_HELPERFUNCS::displayBuildLog(program, device);
    CL_HELPERFUNCS::checkErr(err, "Build::Build()");
    
    
    
    //====================================
    //set kernel entry point
    std::string kernelFunc_fep_comb = "gf_fep_comb_GPU";
    cl::Kernel kernel_fep_comb(program, kernelFunc_fep_comb.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    std::cout << __FANCY__ << "Entry point set to " << kernelFunc_fep_comb << std::endl;
    
    //====================================
    //set kernel entry point
    std::string kernelFunc_fep_set = "gf_fep_set_GPU";
    cl::Kernel kernel_fep_set(program, kernelFunc_fep_set.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    std::cout << __FANCY__ << "Entry point set to " << kernelFunc_fep_set << std::endl;

    //====================================
    //set kernel entry point
    std::string kernelFunc_kFit = "kFit";
    cl::Kernel kernel_kFit(program, kernelFunc_kFit.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    std::cout << __FANCY__ << "Entry point set to " << kernelFunc_kFit << std::endl;
    /*
    //====================================
    //set kernel entry point
    std::string kernelFunc_fit_format = "gf_fit_format_GPU";
    cl::Kernel kernel_fep_set(program, kernelFunc_fep_set.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    std::cout << __FANCY__ << "Entry point set to " << kernelFunc_fep_set << std::endl;
    */
    // Get the maximum work group size for executing the kernel on the device
    
    cl::NDRange maxWorkGroupSize;
    err = kernel_fep_comb.getWorkGroupInfo(device(),
				  CL_KERNEL_WORK_GROUP_SIZE,
				  &maxWorkGroupSize); //returns 3 dimensional size
    
    std::cout << __FANCY__ << "Max work group size = [" << 
      maxWorkGroupSize[0] << "," << 
      maxWorkGroupSize[1] << "," <<
      maxWorkGroupSize[2] << "]" << std::endl;
    //		 if(N_THREADS_PER_BLOCK > maxWorkGroupSize[0]) N_THREADS_PER_BLOCK = maxWorkGroupSize[0];
    //std::cout << __FANCY__ << "Work group size(N_THREADS_PER_BLOCK) = " << N_THREADS_PER_BLOCK << std::endl;
    
    avgTime = 0;
    
    std::cout << "device type " << CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) << std::endl;


    cl::Buffer evt_CL(
		    context,
		    CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
		    0:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		    sizeof(evt_arrays),
		    evt,
		    &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() IN");
    
    cl::Buffer fit_dev_CL(
			  context,
			  CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
			  CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			  sizeof(fit_arrays),
			  fit_dev,
			  &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT");

    cl::Buffer fep_dev_CL(
		     context,
		     CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
		     CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		     sizeof(fep_arrays),
		     fep_dev,
		     &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT");
    
    
    cl::Buffer edata_dev_CL(
			    context,
			    CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
			    CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			    sizeof(extra_data),
			    edata_dev,
			    &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT");
    
    gettimeofday(&ptBegin, NULL);
    gf_fep_unpack_evt(evt, k, data_send);
    printf("Total events %d\n\n",evt->totEvts);
    gettimeofday(&ptEnd, NULL);
    printf("Time to CPU unpack: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    err = kernel_fep_comb.setArg(0, evt_CL);
    err = kernel_fep_comb.setArg(1, fep_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    printf("We have prepared the buffers and are ready to go!\n");
    err = kernel_fep_set.setArg(0, evt_CL);
    err = kernel_fep_set.setArg(1, fep_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_kFit.setArg(0, fep_dev_CL);
    err = kernel_kFit.setArg(1, edata_dev_CL);
    err = kernel_kFit.setArg(2, fit_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    printf("We have prepared the buffers and are ready to go!\n");

    printf("Size of arrays is evt_arrays=%d, fep_arrays=%d, fit_arrays=%d, extra_data=%d\n",
	   sizeof(evt_arrays),sizeof(fep_arrays),sizeof(fit_arrays), sizeof(extra_data));

    gettimeofday(&ptBegin, NULL);
    err = queue.enqueueWriteBuffer(
				   evt_CL,
				   CL_TRUE,
				   0,
				   sizeof(evt_arrays),
				   evt);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueWriteBuffer()");
    
    err = queue.enqueueNDRangeKernel(
				     kernel_fep_comb,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXROAD),
				     cl::NDRange(MAXROAD),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");

    event.wait();


    err = queue.enqueueNDRangeKernel(
				     kernel_fep_set,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXCOMB,MAXROAD),
				     cl::NDRange(MAXCOMB,1),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");

    event.wait();

    err = queue.finish();
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::clFinish1()");
        
    err = queue.enqueueNDRangeKernel(
				     kernel_kFit,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXCOMB,MAXROAD*NFITTER),
				     cl::NDRange(MAXCOMB,NFITTER),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel()");

    event.wait();

    err = queue.finish();
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::clFinish2()");

    err = queue.enqueueReadBuffer(
				  fep_dev_CL,
				  CL_TRUE,
				  0,
				  sizeof(fep_arrays),
				  fep_dev);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueReadBuffer()");

    
    err = queue.enqueueReadBuffer(
				  fit_dev_CL,
				  CL_TRUE,
				  0,
				  sizeof(fit_arrays),
				  fit_dev);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueReadBuffer()");

    err = queue.enqueueReadBuffer(
				  edata_dev_CL,
				  CL_TRUE,
				  0,
				  sizeof(extra_data),
				  edata_dev);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueReadBuffer()");

    //printf("We made it here (1)...\n");
    //printf("fep_dev = %p\n", fep_dev);
    //		     rdtscl(start);
    /*
    err = queue.enqueueReadBuffer(
				  fep_dev_CL,
				  CL_TRUE,
				  0,
				  sizeof(fep_arrays),
				  fep_dev);
    //printf("We made it here (2)...\n");
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueReadBuffer()");
    //		     rdtscl(start);
    */
    gettimeofday(&ptEnd, NULL);
    printf("Time to do combinations, OpenCL: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    for(int ie=0; ie<NEVTS; ie++){
      printf("\nEvent %d, nroads = %d, fep_err_sum=%d, fit_err_sum=%d",ie,fep_dev->fep_nroads[ie],fep_dev->fep_err_sum[ie],fit_dev->fit_err_sum[ie]);
      for(int ir=0; ir<MAXROAD; ir++){
	if(fep_dev->fep_ncmb[ie][ir]!=0)
	  printf("\n\tRoad %d, ncomb = %d",ir,fep_dev->fep_ncmb[ie][ir]);
	for(int ic=0; ic<fep_dev->fep_ncmb[ie][ir]; ic++){
	  printf("\n\t\t ncomb5h=%d",fep_dev->fep_ncomb5h[ie][ir][ic]);
	}
      }
    }


    set_outcable(tf);   
  }
  gettimeofday(&tEnd, NULL);
  printf("Time to complete all: %.3f ms\n",
          ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0);

  // write output file
  FILE* OUTCHECK = fopen(fileOut, "w");

  for (int i=0; i< tf->out->ndata; i++)
    fprintf(OUTCHECK,"%.6x\n", tf->out->data[i]);

  fclose(OUTCHECK);
  
  return 0;
}
