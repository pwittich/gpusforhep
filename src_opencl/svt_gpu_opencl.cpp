#include <unistd.h>
#include <sys/time.h>
#include "semaphore.c"
#include "svt_gpu_opencl.h"
#include <math.h>

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include <CL/cl.hpp>
#include "helperFuncs.h"
#include "clHelperFuncs.h"
#include <string>
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <malloc.h>

const int gf_maskdata[] = {
  0x00000000,
  0x00000001, 0x00000003, 0x00000007, 0x0000000f,
  0x0000001f, 0x0000003f, 0x0000007f, 0x000000ff,
  0x000001ff, 0x000003ff, 0x000007ff, 0x00000fff,
  0x00001fff, 0x00003fff, 0x00007fff, 0x0000ffff,
  0x0001ffff, 0x0003ffff, 0x0007ffff, 0x000fffff,
  0x001fffff, 0x003fffff, 0x007fffff, 0x00ffffff,
  0x01ffffff, 0x03ffffff, 0x07ffffff, 0x0fffffff,
  0x1fffffff, 0x3fffffff, 0x7fffffff, 0xffffffff
};

const unsigned long gf_maskdata3[] = {
  0x000000000000UL,
  0x000000000001UL, 0x000000000003UL, 0x000000000007UL, 0x00000000000fUL,
  0x00000000001fUL, 0x00000000003fUL, 0x00000000007fUL, 0x0000000000ffUL,
  0x0000000001ffUL, 0x0000000003ffUL, 0x0000000007ffUL, 0x000000000fffUL,
  0x000000001fffUL, 0x000000003fffUL, 0x000000007fffUL, 0x00000000ffffUL,
  0x00000001ffffUL, 0x00000003ffffUL, 0x00000007ffffUL, 0x0000000fffffUL,
  0x0000001fffffUL, 0x0000003fffffUL, 0x0000007fffffUL, 0x000000ffffffUL,
  0x000001ffffffUL, 0x000003ffffffUL, 0x000007ffffffUL, 0x00000fffffffUL,
  0x00001fffffffUL, 0x00003fffffffUL, 0x00007fffffffUL, 0x0000ffffffffUL,
  0x0001ffffffffUL, 0x0003ffffffffUL, 0x0007ffffffffUL, 0x000fffffffffUL,
  0x001fffffffffUL, 0x003fffffffffUL, 0x007fffffffffUL, 0x00ffffffffffUL,
  0x01ffffffffffUL, 0x03ffffffffffUL, 0x07ffffffffffUL, 0x0fffffffffffUL,
  0x1fffffffffffUL, 0x3fffffffffffUL, 0x7fffffffffffUL, 0xffffffffffffUL 
};

typedef unsigned int	   DATAWORD; //e.g. UINT32 or float //NOTE: this must absolutely match the typedef in kernel!!!!

/*
__kernel void init_arrays_GPU (fout_arrays* fout_dev, evt_arrays* evt_dev, int* events ) {    
}
*/

void svt_GPU(tf_arrays_t tf, unsigned int *data_in, int n_words) {

}


void set_outcable_fout(fout_arrays* fout_dev, int totEvts, unsigned int *&data_rec, int &ow) {

  svtsim_cable_t *out;
  out = svtsim_cable_new();

  svtsim_cable_copywords(out, 0, 0);

  for (int ie=0; ie < totEvts; ie++) {
    for (int nt=0; nt < fout_dev->fout_ntrks[ie]; nt++) {
      svtsim_cable_addwords(out, fout_dev->fout_gfword[ie][nt], NTFWORDS);
    }
    svtsim_cable_addword(out, fout_dev->fout_ee_word[ie]);
  }

  ow = out->ndata;

  for (int i=0; i < ow ; i++) {
    data_rec[i] = out->data[i];
    //printf("output word %d is 0x%x\n",i,out->data[i]);
  }

//  svtsim_free(out);
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

  unsigned int *data_rec = (unsigned int*)malloc(2500000*sizeof(unsigned));
  int ow;


  std::cout << "get_page_size return is " << getpagesize() << std::endl;

  int totEvts;
  
  struct evt_arrays *evt = new evt_arrays;
  struct fep_arrays *fep_dev = new fep_arrays;
  struct extra_data *edata_dev = new extra_data;
  struct fit_arrays *fit_dev = new fit_arrays;
  struct fout_arrays *fout_dev = new fout_arrays;
  
  int n_words=k;
  
  int N_THREADS_PER_BLOCK = 32;
  int *ids, *out1, *out2, *out3;
  int tEvts = 0;
  // unsigned int *d_data_in;
  long sizeW = sizeof(int) * n_words;
  
  ids  = (int *)malloc(sizeW);
  out1 = (int *)malloc(sizeW);
  out2 = (int *)malloc(sizeW);
  out3 = (int *)malloc(sizeW);
  
  /*
  struct evt_arrays *evt = (struct evt_arrays*)memalign(getpagesize(),sizeof(struct evt_arrays));
  struct fep_arrays *fep_dev = (struct fep_arrays*)memalign(getpagesize(),sizeof(struct fep_arrays));
  struct extra_data *edata_dev = (struct extra_data*)memalign(getpagesize(),sizeof(struct extra_data));
  struct fit_arrays *fit_dev = (struct fit_arrays*)memalign(getpagesize(),sizeof(struct fit_arrays));
  struct fout_arrays *fout_dev = (struct fout_arrays*)memalign(getpagesize(),sizeof(struct fout_arrays));
  */
  /*
  struct evt_arrays *evt = (struct evt_arrays*)memalign(pow(2,ceil(log(sizeof(struct evt_arrays))/log(2))),sizeof(struct evt_arrays));
  struct fep_arrays *fep_dev = (struct fep_arrays*)memalign(pow(2,ceil(log(sizeof(struct fep_arrays))/log(2))),sizeof(struct fep_arrays));
  struct extra_data *edata_dev = (struct extra_data*)memalign(pow(2,ceil(log(sizeof(struct extra_data))/log(2))),sizeof(struct extra_data));
  struct fit_arrays *fit_dev = (struct fit_arrays*)memalign(pow(2,ceil(log(sizeof(struct fit_arrays))/log(2))),sizeof(struct fit_arrays));
  struct fout_arrays *fout_dev = (struct fout_arrays*)memalign(pow(2,ceil(log(sizeof(struct fout_arrays))/log(2))),sizeof(struct fout_arrays));
  */

  svtsim_fconread(tf, edata_dev);

  //free(tf);

  bool is_null=false;

  if(evt==NULL) is_null=true;

  printf("Is evt_arrays ptr null? bool=%d \n",is_null);

  //gf_init_evt(&evt);

  gettimeofday(&tBegin, NULL);

  const bool PRINT_TIME=true;
  const int N_LOOPS=10;
  const int N_CHECKS=5;
  float times[N_CHECKS][N_LOOPS];
  int n_iters=0;

  /*
  if ( strcmp(where,"cpu") == 0 ) { // CPU
    memcpy(tf->wedge, edata_dev->wedge, sizeof(edata_dev->wedge));
    memcpy(tf->whichFit, edata_dev->whichFit, sizeof(edata_dev->whichFit));
    memcpy(tf->lfitparfcon, edata_dev->lfitparfcon, sizeof(edata_dev->lfitparfcon));
    printf("Start work on CPU..... \n");
  */

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
  
/*
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

    for(int ie=0; ie<NEVTS; ie++)
      printf("Event %d, evt_nroads = %d, fep_nroads=%d, fit_err_sum=%d, fout_ntrks=%d, fout_parity=%d\n",ie,tf->evt_nroads[ie],tf->fep_nroads[ie],tf->fit_err_sum[ie],tf->fout_ntrks[ie],tf->fout_parity[ie]);

    printf(".... fits %d events! \n", tf->totEvts);

    ow = tf->out->ndata;
    memcpy(data_rec, tf->out->data, sizeof(int)*tf->out->ndata);


      } else { // GPU
    */
  //HELPERFUNCS::TemplateDataType<DATAWORD> dataType;

    //get OpenCL platforms and their devices in a vector
    //std::cout << __FANCY__ << "getting OpenCL cable platforms and their devices in a vector..." << std::endl;
    
    cl_int err;
    cl::vector< cl::Platform > platformList;
    cl::vector< cl::vector< cl::Device > *> deviceList;
    unsigned int dev_i, plat_i;
    unsigned int start, finish;
    float avgTime;
    cl::Event event;
    
    CL_HELPERFUNCS::getPlatformsAndDevices(&platformList,&deviceList,dev_i,plat_i);

    //this is for GTX 285 and 590...
    //dev_i = 0;
    //plat_i = 1;	
    
    //this is for AMD card...
    dev_i = 0;
    plat_i = 0;	


    HELPERFUNCS::TemplateDataType<DATAWORD> dataType;

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
    std::string kernel_file_init = "gf_init.cl";
    std::ifstream file_init((kernel_file_init).c_str());
    CL_HELPERFUNCS::checkErr(file_init.is_open() ? CL_SUCCESS:-1, (kernel_file_init).c_str());
    
    
    //TODO-RAR add build options
    std::string prog_init(std::istreambuf_iterator<char>(file_init),
		     (std::istreambuf_iterator<char>()));
    std::string buildOptions_init;
    { // create preprocessor defines for the kernel
      char buf[256]; 
      //sprintf(buf,"-cl-mad-enable -D DATAWORD=%s ", dataType.getType().c_str());
      sprintf(buf,"-cl-mad-enable -I./");
      buildOptions_init = buf;
      //std::cout << __FANCY__ << "buildOptions = " << buildOptions_init << std::endl;
    }
    
    cl::Program::Sources source_init(1,std::make_pair(prog_init.c_str(), prog_init.length()));
    
    cl::Program program_init(context, source_init);
    
    err = program_init.build(*(deviceList[plat_i]),buildOptions_init.c_str());
	      
    CL_HELPERFUNCS::displayBuildLog(program_init, device);
    CL_HELPERFUNCS::checkErr(err, "Build::Build()");

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
      //sprintf(buf,"-cl-mad-enable -D DATAWORD=%s ", dataType.getType().c_str());
      sprintf(buf,"-cl-mad-enable -I./");
      buildOptions = buf;
      //std::cout << __FANCY__ << "buildOptions = " << buildOptions << std::endl;
    }
    
    cl::Program::Sources source(1,std::make_pair(prog.c_str(), prog.length()));
    
    cl::Program program(context, source);
    
    err = program.build(*(deviceList[plat_i]),buildOptions.c_str());
	      
    CL_HELPERFUNCS::displayBuildLog(program, device);
    CL_HELPERFUNCS::checkErr(err, "Build::Build()");
    
    
    
    //====================================
    //set kernel entry point
    std::string kernelFunc_init = "init_arrays_GPU";
    cl::Kernel kernel_init(program_init, kernelFunc_init.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_init << std::endl;

    //====================================
    //set kernel entry point
    std::string kernelFunc_unpack = "k_word_decode";
    cl::Kernel kernel_unpack(program, kernelFunc_unpack.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_fep_comb << std::endl;
 
   //====================================
    //set kernel entry point
    std::string kernelFunc_fep_comb = "gf_fep_comb_GPU";
    cl::Kernel kernel_fep_comb(program, kernelFunc_fep_comb.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_fep_comb << std::endl;
    
    //====================================
    //set kernel entry point
    std::string kernelFunc_fep_set = "gf_fep_set_GPU";
    cl::Kernel kernel_fep_set(program, kernelFunc_fep_set.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_fep_set << std::endl;

    //====================================
    //build program source
    std::string kernel_fit_file = "gf_fit.cl";
    std::ifstream fit_file((kernel_fit_file).c_str());
    CL_HELPERFUNCS::checkErr(fit_file.is_open() ? CL_SUCCESS:-1, (kernel_fit_file).c_str());
    
    
    //printf("And now build the other piece/n");

    //TODO-RAR add build options
    std::string fit_prog(std::istreambuf_iterator<char>(fit_file),
			 (std::istreambuf_iterator<char>()));
    std::string fit_buildOptions;
    { // create preprocessor defines for the kernel
      char fit_buf[256]; 
      //sprintf(fit_buf,"-cl-mad-enable -D DATAWORD=%s ", dataType.getType().c_str());
      sprintf(fit_buf,"-cl-mad-enable -I./");
      fit_buildOptions = fit_buf;
      //std::cout << __FANCY__ << "fit_buildOptions = " << fit_buildOptions << std::endl;
    }
    
    cl::Program::Sources fit_source(1,std::make_pair(fit_prog.c_str(), fit_prog.length()));
    
    cl::Program fit_program(context, fit_source);
    
    err = fit_program.build(*(deviceList[plat_i]),buildOptions.c_str());
	      
    //CL_HELPERFUNCS::displayBuildLog(fit_program, device);
    CL_HELPERFUNCS::checkErr(err, "Build::Build()");

    //====================================
    //set kernel entry point
    std::string kernelFunc_kFit = "kFit";
    cl::Kernel kernel_kFit(fit_program, kernelFunc_kFit.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_kFit << std::endl;

    //====================================
    //set kernel entry point
    std::string kernelFunc_fit_format = "gf_fit_format_GPU";
    cl::Kernel kernel_fit_format(fit_program, kernelFunc_fit_format.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_fit_format << std::endl;

    //====================================
    //set kernel entry point
    std::string kernelFunc_comparator = "gf_comparator_GPU";
    cl::Kernel kernel_comparator(fit_program, kernelFunc_comparator.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_comparator << std::endl;

    //====================================
    //set kernel entry point
    std::string kernelFunc_compute_eeword = "gf_compute_eeword_GPU";
    cl::Kernel kernel_compute_eeword(fit_program, kernelFunc_compute_eeword.c_str(), &err);
    CL_HELPERFUNCS::checkErr(err, "Kernel::Kernel()");
    //std::cout << __FANCY__ << "Entry point set to " << kernelFunc_compute_eeword << std::endl;

    // Get the maximum work group size for executing the kernel on the device
    
    cl::NDRange maxWorkGroupSize;
    err = kernel_fep_comb.getWorkGroupInfo(device(),
				  CL_KERNEL_WORK_GROUP_SIZE,
				  &maxWorkGroupSize); //returns 3 dimensional size
    
    //std::cout << __FANCY__ << "Max work group size = [" << 
    //maxWorkGroupSize[0] << "," << 
    //maxWorkGroupSize[1] << "," <<
    //maxWorkGroupSize[2] << "]" << std::endl;
    //		 if(N_THREADS_PER_BLOCK > maxWorkGroupSize[0]) N_THREADS_PER_BLOCK = maxWorkGroupSize[0];
    ////std::cout << __FANCY__ << "Work group size(N_THREADS_PER_BLOCK) = " << N_THREADS_PER_BLOCK << std::endl;
    
    avgTime = 0;
    
    //std::cout << "device type " << CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) << std::endl;


    printf("Size of arrays is evt_arrays=%d, fep_arrays=%d, fit_arrays=%d, extra_data=%d, fout_arrays=%d\n",
	   sizeof(evt_arrays),sizeof(fep_arrays),sizeof(fit_arrays), sizeof(extra_data),sizeof(fout_arrays));


    cl::Buffer edata_dev_CL(
			    context,
			    CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
			    0:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			    sizeof(extra_data),
			    CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
			    NULL:edata_dev,
			    &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() IN");

    err = queue.enqueueWriteBuffer(
				   edata_dev_CL,
				   CL_TRUE,
				   0,
				   sizeof(extra_data),
				   edata_dev);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueWriteBuffer(edata)");

    err = queue.finish();

  while (n_iters < N_LOOPS){
    if(n_iters%10==0) printf("Processing loop %d\n",n_iters);
    
    cl::Buffer input_CL(
		      context,
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      0:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      k*sizeof(int),
		      //evt,
		      NULL,
		      &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() IN");
    
    cl::Buffer ids_CL(
		      context,
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      k*sizeof(int),
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      NULL:ids,
		      &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() IDS");

    cl::Buffer out1_CL(
		      context,
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      k*sizeof(int),
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      NULL:out1,
		      &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT1");

    cl::Buffer out2_CL(
		      context,
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      k*sizeof(int),
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      NULL:out2,
		      &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT2");

    cl::Buffer out3_CL(
		      context,
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      k*sizeof(int),
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      NULL:out3,
		      &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT3");
    

    cl::Buffer fout_dev_CL(
			    context,
			    CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
			    CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			    sizeof(fout_arrays),
			    (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
			    NULL:fout_dev,
			    &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT (fout_arrays)");

    cl::Buffer evt_CL(
		      context,
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      0:CL_MEM_USE_HOST_PTR, //CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		      sizeof(evt_arrays),
		      (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
		      NULL:evt,
		      &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() IN");

    
    cl::Buffer fit_dev_CL(
			  context,
			  CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
			  CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			  sizeof(fit_arrays),
			  (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
			  NULL:fit_dev,
			  &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT (fit_dev)");
    
    //printf("Size of arrays is evt_arrays=%d, fep_arrays=%d, fit_arrays=%d, extra_data=%d\n",
    //	   sizeof(evt_arrays),sizeof(fep_arrays),sizeof(fit_arrays), sizeof(extra_data));

    cl::Buffer fep_dev_CL(
			  context,
			  CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i) ?
			  CL_MEM_READ_WRITE:CL_MEM_USE_HOST_PTR, //CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			  sizeof(fep_arrays),
			  (CL_HELPERFUNCS::isDeviceTypeGPU(&deviceList,plat_i,dev_i)==1) ?
			  NULL:fep_dev,
			  &err);
    CL_HELPERFUNCS::checkErr(err, "Buffer::Buffer() OUT (fep_dev)");
        
    err = kernel_init.setArg(1, evt_CL);
    err = kernel_init.setArg(0, fout_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");

    err = kernel_unpack.setArg(0,n_words);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_unpack.setArg(1,input_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_unpack.setArg(2,ids_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_unpack.setArg(3,out1_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_unpack.setArg(4,out2_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_unpack.setArg(5,out3_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");

    err = kernel_fep_comb.setArg(0, evt_CL);
    err = kernel_fep_comb.setArg(1, fep_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    //printf("We have prepared the buffers and are ready to go!\n");
    err = kernel_fep_set.setArg(0, evt_CL);
    err = kernel_fep_set.setArg(1, fep_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_kFit.setArg(0, fep_dev_CL);
    err = kernel_kFit.setArg(1, edata_dev_CL);
    err = kernel_kFit.setArg(2, fit_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_fit_format.setArg(0, fep_dev_CL);
    err = kernel_fit_format.setArg(1, fit_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_comparator.setArg(0, fep_dev_CL);
    err = kernel_comparator.setArg(1, evt_CL);
    err = kernel_comparator.setArg(2, fit_dev_CL);
    err = kernel_comparator.setArg(3, fout_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");
    err = kernel_compute_eeword.setArg(0, fep_dev_CL);
    err = kernel_compute_eeword.setArg(1, fit_dev_CL);
    err = kernel_compute_eeword.setArg(2, fout_dev_CL);
    CL_HELPERFUNCS::checkErr(err, "Kernel::setArg()");

    
    err = queue.enqueueNDRangeKernel(
    				     kernel_init,
				     cl::NullRange,
				     cl::NDRange(NEVTS*(NSVX_PLANE+1),MAXROAD),
				     cl::NDRange(NSVX_PLANE+1,1),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(init)");

    err = queue.finish();
    
    //printf("We have prepared the buffers and are ready to go!\n");

    //printf("Size of arrays is evt_arrays=%d, fep_arrays=%d, fit_arrays=%d, extra_data=%d\n",
    //	   sizeof(evt_arrays),sizeof(fep_arrays),sizeof(fit_arrays), sizeof(extra_data));
    
    gettimeofday(&ptBegin, NULL);
    /*
    gf_fep_unpack_evt(evt, k, data_send); //printf("Total events %d\n\n",evt->totEvts);

    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to CPU unpack: %.3f ms\n",
			  ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[0][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
    */
    
    
    err = queue.enqueueWriteBuffer(
				   input_CL,
				   CL_TRUE,
				   0,
				   k*sizeof(int),
				   data_send);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueWriteBuffer()");

    std::cout << "Wrote input to GPU" << std::endl;

    err = queue.enqueueNDRangeKernel(
				     kernel_unpack,
				     cl::NullRange,
				     cl::NDRange((n_words+N_THREADS_PER_BLOCK-1)),
				     cl::NDRange(N_THREADS_PER_BLOCK),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(kernel_unpack)");

    std::cout << "Ran kernel on GPU" << std::endl;

    err = queue.enqueueReadBuffer(
				  out1_CL,
				  CL_TRUE,
				  0,
				  k*sizeof(int),
				  out1);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueReadBuffer(out1)");

    err = queue.enqueueReadBuffer(
				  out2_CL,
				  CL_TRUE,
				  0,
				  k*sizeof(int),
				  out2);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueReadBuffer(out2)");

    err = queue.enqueueReadBuffer(
				  out3_CL,
				  CL_TRUE,
				  0,
				  k*sizeof(int),
				  out3);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueReadBuffer(out3)");

    err = queue.enqueueReadBuffer(
				  ids_CL,
				  CL_TRUE,
				  0,
				  k*sizeof(int),
				  ids);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueReadBuffer(ids)");

    //std::cout << "Got output from GPU" << std::endl;

    memset(evt->evt_nroads, 0, sizeof(evt->evt_nroads));
    memset(evt->evt_err_sum, 0, sizeof(evt->evt_err_sum));
    memset(evt->evt_layerZ, 0, sizeof(evt->evt_layerZ));
    memset(evt->evt_nhits, 0,  sizeof(evt->evt_nhits));
    memset(evt->evt_err,  0,   sizeof(evt->evt_err));
    memset(evt->evt_zid,  0,   sizeof(evt->evt_zid));

    //std::cout << "Did memset" << std::endl;

    for (int ie = 0; ie < NEVTS; ie++) {
      evt->evt_zid[ie][evt->evt_nroads[ie]] = -1; // because we set it to 0 for GPU version
    }
    
    //std::cout << "Did some more event pulling" << std::endl;
    
    int id_last = -1;
    int my_event = EVT;
    int id;
    
    for (int i = 0; i < n_words; i++) {
      
      //std::cout << "Inside nwords loop on " << i << " of " << n_words << std::endl;

      id = ids[i];
      
      //std::cout << "\tGot id" << std::endl;

      bool gf_xft = 0;
      if (id == XFT_LYR_2) { // compatibility - stp
	id = XFT_LYR;
	gf_xft = 1;
      }
      
      //std::cout << "\tGot xft thing" << std::endl;

      //std::cout << "\tMy event is " << my_event << std::endl;

      int nroads = evt->evt_nroads[my_event];
      //std::cout << "\tGot roads: " << nroads << std::endl;

      int nhits = evt->evt_nhits[my_event][nroads][id];

      //std::cout << "\tGot roads and " << nhits << " hits" << std::endl;
      
      // SVX Data
      if (id < XFT_LYR) {
	int zid = out1[i];
	int lcl = out2[i];
	int hit = out3[i];
	
	evt->evt_hit[my_event][nroads][id][nhits] = hit;
	evt->evt_hitZ[my_event][nroads][id][nhits] = zid;
	evt->evt_lcl[my_event][nroads][id][nhits] = lcl;
	evt->evt_lclforcut[my_event][nroads][id][nhits] = lcl;
	evt->evt_layerZ[my_event][nroads][id] = zid;
	
	if (evt->evt_zid[my_event][nroads] == -1) {
	  evt->evt_zid[my_event][nroads] = zid & gf_maskdata[GF_SUBZ_WIDTH];
	} else {
	  evt->evt_zid[my_event][nroads] = (((zid & gf_maskdata[GF_SUBZ_WIDTH]) << GF_SUBZ_WIDTH)
					    + (evt->evt_zid[my_event][nroads] & gf_mask(GF_SUBZ_WIDTH)));
	}
	
	nhits = ++evt->evt_nhits[my_event][nroads][id];
	
	// Error Checking
	if (nhits == MAX_HIT) evt->evt_err[my_event][nroads] |= (1 << OFLOW_HIT_BIT);
	if (id < id_last) evt->evt_err[my_event][nroads] |= (1 << OUTORDER_BIT);
      } else if (id == XFT_LYR && gf_xft == 0) {
	// we ignore - stp
      } else if (id == XFT_LYR && gf_xft == 1) {
	int crv = out1[i];
	int crv_sign = out2[i];
	int phi = out3[i];

	evt->evt_crv[my_event][nroads][nhits] = crv;
	evt->evt_crv_sign[my_event][nroads][nhits] = crv_sign;
	evt->evt_phi[my_event][nroads][nhits] = phi;

	nhits = ++evt->evt_nhits[my_event][nroads][id];

	// Error Checking
	if (nhits == MAX_HIT) evt->evt_err[my_event][nroads] |= (1 << OFLOW_HIT_BIT);
	if (id < id_last) evt->evt_err[my_event][nroads] |= (1 << OUTORDER_BIT);
      } else if (id == EP_LYR) {
	int sector = out1[i];
	int amroad = out2[i];

	evt->evt_cable_sect[my_event][nroads] = sector;
	evt->evt_sect[my_event][nroads] = sector;
	evt->evt_road[my_event][nroads] = amroad;
	evt->evt_err_sum[my_event] |= evt->evt_err[my_event][nroads];

	nroads = ++evt->evt_nroads[my_event];

	if (nroads > MAXROAD) {
	  printf("The limit on the number of roads fitted by the TF is %d\n",MAXROAD);
	  printf("You reached that limit evt->nroads = %d\n",nroads);
	}

	for (id = 0; id <= XFT_LYR; id++)
	  evt->evt_nhits[my_event][nroads][id] = 0;

	evt->evt_err[my_event][nroads] = 0;
	evt->evt_zid[my_event][nroads] = -1;

	id = -1; id_last = -1;
      } else if (id == EE_LYR) {

	/*
	std::cout << "END OF EVENT!" << std::endl;
	std::cout << "\tInside nwords loop on " << i << " of " << n_words << std::endl;
	std::cout << "\tMy event is " << my_event << std::endl;


	int my_roads = evt->evt_nroads[my_event];
	int my_hits=0;
	for(int ir=0; ir<my_roads; ir++){
	  for(int il=0; il<NSVX_PLANE+1; il++){
	    my_hits += evt->evt_nhits[my_event][ir][il];	  
	  }
	}
	std::cout << "\tGot roads: " << my_roads << std::endl;
	std::cout << "\tGot hits: " << my_hits << std::endl;
	*/
	evt->evt_ee_word[my_event] = out1[i];
	tEvts++;
	my_event++;


	id = -1; id_last = -1;
      } else {
	printf("Error INV_DATA_BIT: layer = %u\n", id);
	evt->evt_err[my_event][nroads] |= (1 << INV_DATA_BIT);
      }
      id_last = id;

      //std::cout << "\tEnd of loop" << std::endl;

    } //end loop on input words

    
    //std::cout << "Finished cpu part" << std::endl;


    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to GPU unpack: %.3f ms\n",
			  ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[0][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

    gettimeofday(&ptBegin, NULL);
    /*
    err = queue.enqueueNDRangeKernel(
    				     kernel_init,
				     cl::NullRange,
				     cl::NDRange(NEVTS*(NSVX_PLANE+1),MAXROAD),
				     cl::NDRange(NSVX_PLANE+1,1),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(init)");

    err = queue.finish();
    */
    err = queue.enqueueWriteBuffer(
				   evt_CL,
				   CL_TRUE,
				   0,
				   sizeof(evt_arrays),
				   evt);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueWriteBuffer(evt)");
   
    err = queue.finish();

    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to GPU copy: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[1][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

    gettimeofday(&ptBegin, NULL);

    err = queue.enqueueNDRangeKernel(
				     kernel_fep_comb,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXROAD),
				     cl::NDRange(MAXROAD),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(fep_comb)");

    // event.wait();

    /*
    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to find combinations: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[2][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

    gettimeofday(&ptBegin, NULL);
    */
    err = queue.enqueueNDRangeKernel(
				     kernel_fep_set,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXCOMB,MAXROAD),
				     cl::NDRange(MAXCOMB,1),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(fep_set)");

    //event.wait();

    //err = queue.finish();
    //CL_HELPERFUNCS::checkErr(err, "ComamndQueue::clFinish1()");
        
    err = queue.finish();
    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to setup fep arrays: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[2][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

    gettimeofday(&ptBegin, NULL);

    err = queue.enqueueNDRangeKernel(
				     kernel_kFit,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXCOMB,MAXROAD*NFITTER),
				     cl::NDRange(MAXCOMB,NFITTER),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(kFit)");

    //event.wait();
    /*
    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to do fit: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[4][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
    //gettimeofday(&ptBegin, NULL);
    */

    err = queue.enqueueNDRangeKernel(
				     kernel_fit_format,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXCOMB,MAXROAD*MAXCOMB5H),
				     cl::NDRange(MAXCOMB,MAXCOMB5H),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(fit_format)");

    //event.wait();
    /*
    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to format fit: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[5][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
    
    gettimeofday(&ptBegin, NULL);
    */
    err = queue.enqueueNDRangeKernel(
				     kernel_comparator,
				     cl::NullRange,
				     cl::NDRange(NEVTS*MAXCOMB,MAXROAD),
				     cl::NDRange(MAXCOMB,1),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(comparator)");

    //event.wait();
    /*
    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to do comparator: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[6][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
    
    gettimeofday(&ptBegin, NULL);
    */
    err = queue.enqueueNDRangeKernel(
				     kernel_compute_eeword,
				     cl::NullRange,
				     //cl::NDRange(((NEVTS+255)/256) * 256),
				     //cl::NDRange(256),
				     cl::NDRange(NEVTS),
				     cl::NDRange(1),
				     NULL,
				     &event);
    CL_HELPERFUNCS::checkErr(err, "ComamndQueue::enqueueNDRangeKernel(compute_eeword)");

    err = queue.finish();
    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to do ee word computation: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[3][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

    gettimeofday(&ptBegin, NULL);

    //event.wait();

    //printf("Error was ... %d\n",err);
    //err = queue.finish();
    //CL_HELPERFUNCS::checkErr(err, "ComamndQueue::clFinish2()");
    /*
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
    */
    err = queue.enqueueReadBuffer(
				  fout_dev_CL,
				  CL_TRUE,
				  0,
				  sizeof(fout_arrays),
				  fout_dev);
    CL_HELPERFUNCS::checkErr(err, "CommandQueue::enqueueReadBuffer(fout)");

    err = queue.finish();
    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to copy back: %.3f ms\n",
	   ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    times[4][n_iters] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

    gettimeofday(&ptBegin, NULL);

    err = queue.finish();

    gettimeofday(&ptEnd, NULL);
    if(PRINT_TIME) printf("Time to do everything, OpenCL: %.3f ms\n",
			  ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);
    /*
    for(int ie=0; ie<NEVTS; ie++){
      printf("\nEvent %d, evt_nroads = %d, fep_nroads=%d, fit_err_sum=%d, fout_ntrks=%d, fout_parity=%d",ie,evt->evt_nroads[ie],fep_dev->fep_nroads[ie],fit_dev->fit_err_sum[ie],fout_dev->fout_ntrks[ie],fout_dev->fout_parity[ie]);
            
      for(int ir=0; ir<MAXROAD; ir++){
	if(fep_dev->fep_ncmb[ie][ir]!=0)
	  printf("\n\tRoad %d, ncomb = %d",ir,fep_dev->fep_ncmb[ie][ir]);
	for(int ic=0; ic<fep_dev->fep_ncmb[ie][ir]; ic++){
	  printf("\n\t\t hitmap=%d\t",fep_dev->fep_hitmap[ie][ir][ic]);
	  for(int ip=0; ip<NSVX_PLANE; ip++){
	    printf(" %d",fep_dev->fep_hit[ie][ir][ic][ip]);
	  }
	  printf("\t ncom5h=%d",fep_dev->fep_ncomb5h[ie][ir][ic]);
	  for(int ic5=0; ic5< fep_dev->fep_ncomb5h[ie][ir][ic]; ic5++){
	    printf("\n\t\t\t fit_err=%d \n\t\t\t fit_fit=",fit_dev->fit_err[ie][ir][ic][ic5]);
	    for(int ip=0; ip<6; ip++){
	      printf(" %lu",fit_dev->fit_fit[ie][ip][ir][ic][ic5]);
	    }
	  }
	}
      }
    
    }
    */

    set_outcable_fout(fout_dev, NEVTS, data_rec, ow);
    n_iters++;
    /*
    free(ids);
    free(out1);
    free(out2);
    free(out3);
    */
  }

  //gettimeofday(&tEnd, NULL);
  //printf("Time to complete all: %.3f ms\n",
  //      ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0);


  float time_mean[N_CHECKS]; float time_stdev[N_CHECKS];
  for(int i=0; i<N_LOOPS; i++){
    for(int j=0; j<N_CHECKS; j++){
      if(i==0) time_mean[j] =0;
      //printf("\t\tTime check %d, loop %d: mean=%f ms\n",j,i,time_mean[j]);
      time_mean[j] += (times[j][i])/(float)N_LOOPS;
      //printf("\t\tTime check %d, loop %d: mean=%f ms\n",j,i,time_mean[j]);
      }
  }

  for(int i=0; i<N_LOOPS; i++){
    for(int j=0; j<N_CHECKS; j++){
      if(i==0) time_stdev[j] =0;
      time_stdev[j] += (times[j][i]-time_mean[j])*(times[j][i]-time_mean[j])/(float)N_LOOPS;
      //printf("\t\tTime check %d, loop %d: stdev=%f ms\n",j,i,time_stdev[j]);
      }
  }
  for(int j=0; j<N_CHECKS; j++){
    time_stdev[j] = sqrt(time_stdev[j]);
    printf("Time for time check %d was %f with st.dev. %f\n",j,time_mean[j],time_stdev[j]);
  }

  // write output file
  FILE* OUTCHECK = fopen(fileOut, "w");

/*
  for (int i=0; i< tf->out->ndata; i++)
    fprintf(OUTCHECK,"%.6x\n", tf->out->data[i]);
*/

  for (int i=0; i < ow; i++)
    fprintf(OUTCHECK,"%.6x\n", data_rec[i]);

  fclose(OUTCHECK);
  
  delete evt;
  delete fep_dev;
  delete edata_dev;
  delete fit_dev;
  delete fout_dev;

  free(ids);
  free(out1);
  free(out2);
  free(out3);

  return 0;
}
