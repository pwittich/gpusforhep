#include <unistd.h>
#include <sys/time.h>
#include "svt_utils.h"
#include <math.h>

#include <sched.h>
#include "semaphore.c"
#include <thrust/device_vector.h>


// global variables
int VERBOSE = 0;
int TIMER = 0;

// CUDA timer macros
cudaEvent_t c_start, c_stop;

inline void start_time() {
  if ( TIMER ) {
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, 0);
  }
}

inline float stop_time(const char *msg) {
  float elapsedTime = 0;
  if ( TIMER ) { 
    cudaEventRecord(c_stop, 0);
    cudaEventSynchronize(c_stop);
    cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
    if ( VERBOSE )
      printf("Time to %s: %.3f ms\n", msg, elapsedTime);
  }
  return elapsedTime;
}

// calculate mean and stdev on an array of count floats
void get_mean(float *times_array, int count, float *mean, float *stdev) {

  int j;
  float sum = 0;
  float sumsqr = 0;

  *mean = *stdev = 0;

  for (j=0; j < count; j++) {
    sum += times_array[j];
    sumsqr += pow(times_array[j],2);
  }

  *mean = sum/(float)count;

  *stdev = sqrt(abs((sumsqr/(float)count) - pow(*mean,2)));
}

__global__ void init_arrays_GPU (fout_arrays* fout_dev, evt_arrays* evt_dev, int* events ) {

  int ie, ir, ip;

  *events = 0;

  ie = blockIdx.x; // events index
  ir = blockIdx.y; // roads index
  ip = threadIdx.x; // NSVX_PLANE+1

  // initialize evt arrays....
  evt_dev->evt_nroads[ie] = 0;
  evt_dev->evt_ee_word[ie] = 0;
  evt_dev->evt_err_sum[ie] =0;

  evt_dev->evt_zid[ie][ir] = 0;
  evt_dev->evt_err[ie][ir] = 0;
  evt_dev->evt_cable_sect[ie][ir] = 0;
  evt_dev->evt_sect[ie][ir] = 0;
  evt_dev->evt_road[ie][ir] = 0;

  evt_dev->evt_nhits[ie][ir][ip] = 0;

  // initialize fout arrays....
  fout_dev->fout_ntrks[ie] = 0;
  fout_dev->fout_parity[ie] = 0;
  fout_dev->fout_ee_word[ie] = 0;
  fout_dev->fout_err_sum[ie] = 0;
  fout_dev->fout_cdferr[ie] = 0;
  fout_dev->fout_svterr[ie] = 0;
    
}


void setedata_GPU(tf_arrays_t tf, struct extra_data *edata_dev) {

  int len;
  len = SVTSIM_NBAR * FITBLOCK * sizeof(int);
  MY_CUDA_CHECK(cudaMemcpy(edata_dev->whichFit, tf->whichFit, len, cudaMemcpyHostToDevice));
  len = NFITPAR * (DIMSPA+1) * SVTSIM_NBAR * FITBLOCK * sizeof(long long int);
  MY_CUDA_CHECK(cudaMemcpy(edata_dev->lfitparfcon, tf->lfitparfcon, len, cudaMemcpyHostToDevice));
  len = NEVTS * sizeof(int);
  MY_CUDA_CHECK(cudaMemcpy(edata_dev->wedge, tf->wedge, len, cudaMemcpyHostToDevice));

}


int svt_GPU(tf_arrays_t tf, struct extra_data *edata_dev, unsigned int *data_in, int n_words, float *timer, int nothrust, unsigned int* dataout) {

  int tEvts=0;
  int ndata=0;
  dim3 blocks(NEVTS,MAXROAD);

  start_time();
  // Cuda Malloc
  int* d_tEvts;
  MY_CUDA_CHECK(cudaMalloc((void**)&d_tEvts, sizeof(int)));
  int* ndata_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&ndata_dev, sizeof(int)));
  struct evt_arrays* evt_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&evt_dev, sizeof(evt_arrays)));
  struct fep_arrays *fep_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&fep_dev, sizeof(fep_arrays)));
  struct fit_arrays *fit_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&fit_dev, sizeof(fit_arrays)));
  struct fout_arrays *fout_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&fout_dev, sizeof(fout_arrays)));
  unsigned int *dataout_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&dataout_dev, n_words*sizeof(int)));

  // initialize structures
  init_arrays_GPU<<<blocks, NSVX_PLANE+1>>>(fout_dev, evt_dev, d_tEvts);

  if ( nothrust ) { // use pure cuda version of unpack
    
    unsigned int *d_data_in;
    long sizeW = sizeof(int) * n_words;
    cudaMalloc((void **)&d_data_in, sizeW);

    cudaMemcpy(d_data_in, data_in, sizeW, cudaMemcpyHostToDevice);

    timer[0] = stop_time("input copy and initialize");

    start_time();

    gf_unpack_cuda_GPU(d_data_in, n_words, evt_dev, d_tEvts );

    cudaFree(d_data_in);

  } else { // use thrust version of unpack

    thrust::device_vector<unsigned int> d_vec(n_words+1);
    d_vec[0] = 0;
    thrust::copy(data_in, data_in + n_words, d_vec.begin()+1);

    timer[0] = stop_time("input copy and initialize");

    start_time();

    gf_unpack_thrust_GPU(d_vec, n_words, evt_dev, d_tEvts );

  } 

  timer[1] = stop_time("input unpack");

  MY_CUDA_CHECK(cudaMemcpy(&tEvts, d_tEvts, sizeof(int), cudaMemcpyDeviceToHost));
  tf->totEvts = tEvts;
 
  // Fep comb and set
  start_time();  
  gf_fep_GPU( evt_dev, fep_dev, tEvts );
  timer[2] =stop_time("compute fep combinations");

  // Fit and set Fout
  start_time();
  gf_fit_GPU(fep_dev, evt_dev, edata_dev, fit_dev, fout_dev, tEvts, dataout_dev, ndata_dev);
  timer[3] = stop_time("fit data and set output");

  // Output copy DtoH
  start_time();

  MY_CUDA_CHECK(cudaMemcpy(&ndata, ndata_dev, sizeof(int), cudaMemcpyDeviceToHost));
  MY_CUDA_CHECK(cudaMemcpy(dataout, dataout_dev, ndata * sizeof(int), cudaMemcpyDeviceToHost));

  MY_CUDA_CHECK( cudaFree(evt_dev) );
  MY_CUDA_CHECK( cudaFree(fep_dev) );
  MY_CUDA_CHECK( cudaFree(fit_dev) );
  MY_CUDA_CHECK( cudaFree(fout_dev));
  MY_CUDA_CHECK( cudaFree(d_tEvts));
  MY_CUDA_CHECK( cudaFree(ndata_dev));
  MY_CUDA_CHECK( cudaFree(dataout_dev));
  timer[4] = stop_time("copy output (DtoH)");

  return ndata;

}

void help(char* prog) {

  printf("Use %s [-i fileIn] [-o fileOut] [-s cpu || gpu] [-l #loops] [-u] [-v] [-t] [-p priority] [-h] \n\n", prog);
  printf("  -i fileIn       Input file (Default: hbout_w6_100evts).\n");
  printf("  -o fileOut      Output file (Default: gfout.txt).\n");
  printf("  -s cpu || gpu   Switch between CPU or GPU version (Default: gpu).\n");
  printf("  -l loops        Number of executions (Default: 1).\n");
  printf("  -u              Use pure cuda version for unpack (Default: use thrust version).\n");
  printf("  -v              Print verbose messages.\n");
  printf("  -t              Calculate timing.\n");
  printf("  -p priority     Set scheduling priority to <priority> and cpu affinity - you nedd to be ROOT - (Default: disable).\n");
  printf("  -h              This help.\n");

}

int main(int argc, char* argv[]) {

  int c;
  char* fileIn = "hbout_w6_100evts";
  char* fileOut = "gfout.txt";
  char* where = "gpu";
  int N_LOOPS = 1;
  int PRIORITY = 0;
  int NOTHRUST = 0;

  while ( (c = getopt(argc, argv, "i:s:o:l:uvtp:h")) != -1 ) {
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
      case 'l':
        N_LOOPS = atoi(optarg);
        break;
      case 'v':
        VERBOSE = 1;
        break;
      case 'u':
        NOTHRUST = 1;
        break;
      case 't':
        TIMER = 1;
        break;
      case 'p':
        PRIORITY = atoi(optarg);
        break;
      case 'h':
        help(argv[0]);
        return 0;
    }
  }

  if (access(fileIn, 0) == -1) {
    printf("ERROR: File %s doesn't exist.\n", fileIn);
    return 1;
  }

  int semid;
  if ( PRIORITY ) {
    // lock control so no one else can run at the same time and crash the machine
    key_t key = (key_t) 0xdeadface;

    if ((semid = initsem(key, 1)) == -1) {
        perror("initsem");
        exit(1);
    }
    printf("Trying to gain control...\n");
    lock(semid);

    // set scheduling priority & CPU affinity
    struct sched_param p;
    p.sched_priority = PRIORITY;
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
  }

  // Do we want to skip the first "skip" runs from mean calculation?
  int skip = 0;
  int n_iters = N_LOOPS+skip;

  float initg = 0;
  float fcon = 0;
  float timerange = 0;
  float ptime[5];
  float ptime_cpu[3];
  float times_array[6][N_LOOPS];
  float times_array_cpu[4][N_LOOPS];

  struct timeval time_start, time_stop;
  struct timeval tBegin, tEnd;
  struct timeval ptBegin, ptEnd;


  if ( strcmp(where,"gpu") == 0 ) { // GPU

    if ( TIMER ) gettimeofday(&tBegin, NULL);
    
    // this is just to measure time to initialize GPU
    cudaEvent_t     init;
    MY_CUDA_CHECK( cudaEventCreate( &init ) );
    if ( TIMER ) {
      gettimeofday(&tEnd, NULL);
      initg = ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000000.0;  
    }
  }

  
  // read input file
  FILE* hbout = fopen(fileIn,"r");

  if ( hbout == NULL ) {
    printf("ERROR: Cannot open input file\n");
    exit(1);
  }

  unsigned int hexaval;
  unsigned int *data_send = (unsigned int*)malloc(2500000*sizeof(unsigned));
  if ( data_send == (unsigned int*) NULL ) {
    perror("malloc");
    return 2;
  }
  
  char word[16];
  int k=0; // number of words read
  if ( VERBOSE ) printf("Reading input file %s... ", fileIn);
  while (fscanf(hbout, "%s", word) != EOF) {
    hexaval = strtol(word,NULL,16);
    data_send[k] = hexaval;
    k++;
  }

  fclose(hbout);

  int outword;
  unsigned int *dataout = (unsigned int*)malloc(k*sizeof(unsigned));

  tf_arrays_t tf;
  gf_init(&tf);
  svtsim_fconread(tf);

  if ( TIMER ) gettimeofday(&tBegin, NULL);
  
  struct extra_data *edata_dev;

  if ( strcmp(where,"cpu") != 0 ) { // GPU
    if ( TIMER ) start_time();
    MY_CUDA_CHECK(cudaMalloc((void**)&edata_dev, sizeof(struct extra_data)));
    setedata_GPU(tf, edata_dev);
    if ( TIMER ) fcon = stop_time("Copy detector configuration data");
  }

  while (n_iters--) {

    if ( strcmp(where,"cpu") == 0 ) { // CPU
      if ( TIMER ) gettimeofday(&time_start, NULL);
      
      if ( VERBOSE ) printf("Start working on CPU..... \n");
      
      if ( TIMER ) gettimeofday(&ptBegin, NULL);

      gf_fep_unpack(tf, k, data_send);

      if ( TIMER) {
        gettimeofday(&ptEnd, NULL);
        timerange = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
        if ( VERBOSE ) printf("Time to CPU unpack: %.3f ms\n", timerange);
        ptime_cpu[0] = timerange;
 
        gettimeofday(&ptBegin, NULL);
      }
        
      gf_fep_comb(tf);
      
      if ( TIMER) {
        gettimeofday(&ptEnd, NULL);
        timerange = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
        if ( VERBOSE )  printf("Time to CPU comb: %.3f ms\n", timerange);
        ptime_cpu[1] = timerange;

        gettimeofday(&ptBegin, NULL);
      }

      gf_fit(tf);
      gf_comparator(tf);
      
      if ( TIMER) {
        gettimeofday(&ptEnd, NULL);
        timerange = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
        if ( VERBOSE ) printf("Time to CPU fit: %.3f ms\n", timerange);
        ptime_cpu[2] = timerange;
        gettimeofday(&time_stop, NULL); 
      }
      if ( VERBOSE ) printf(".... fits %d events! \n", tf->totEvts);
      

    } else { // GPU
      if ( VERBOSE ) printf("Start working on GPU...\n");
      if ( TIMER ) gettimeofday(&time_start, NULL);  
      outword = svt_GPU(tf, edata_dev, data_send, k, ptime, NOTHRUST, dataout);
      if ( TIMER ) gettimeofday(&time_stop, NULL);
    }

    if ( TIMER ) {
      if ( n_iters < N_LOOPS ) { // skip the first "skip" iterations
        timerange = ((time_stop.tv_usec + 1000000 * time_stop.tv_sec) - (time_start.tv_usec + 1000000 * time_start.tv_sec))/1000.0;
        if ( strcmp(where,"cpu") != 0 ) { // GPU
          times_array[0][n_iters] = timerange;
          for (int t=1; t < 6; ++t) 
            times_array[t][n_iters] = ptime[t-1];
        } else { //CPU
          times_array_cpu[0][n_iters] = timerange; 
          for (int t=1; t < 4; ++t)
            times_array_cpu[t][n_iters] = ptime_cpu[t-1];
        }
      }
    }
  } // end iterations

  if ( strcmp(where,"cpu") != 0 ) {
    MY_CUDA_CHECK(cudaFree(edata_dev));
  }

  if ( TIMER ) {
    gettimeofday(&tEnd, NULL);
    timerange = ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0;
    if ( VERBOSE ) printf("Time to complete all: %.3f ms\n", timerange); 
  }

  // write output file
  FILE* OUTCHECK = fopen(fileOut, "w");
  if ( strcmp(where,"cpu") == 0 )  // CPU
    for (int i=0; i < tf->out->ndata; i++)
      fprintf(OUTCHECK,"%.6x\n", tf->out->data[i]);
  else // GPU
    for (int i=0; i < outword; i++)
      fprintf(OUTCHECK,"%.6x\n", dataout[i]);
  fclose(OUTCHECK);
  
  // write file with times
  if ( TIMER ) {
    char fileTimes[1024];
    FILE *ft;
    if ( strcmp(where,"cpu") != 0 ) { // GPU
      float mean[6];
      float stdev[6];
      for (int t=0; t < 6; ++t) 
        get_mean(times_array[t], N_LOOPS, &mean[t], &stdev[t]);

      sprintf(fileTimes, "ListTimesGPU-Evts_%d_Loops_%d.txt", NEVTS, N_LOOPS);

      ft = fopen(fileTimes, "w");
      fprintf(ft,"# #NEvts: %d, Loops: %d, mean: %.3f ms, stdev: %.3f ms\n", NEVTS, N_LOOPS, mean[0], stdev[0]);
      fprintf(ft,"# initialize GPU: %.3f ms; copy detector configuration data: %.3f ms\n", initg, fcon);
      fprintf(ft,"# input copy and initialize        --> mean: %.3f ms, stdev: %.3f ms\n", mean[1], stdev[1]);
      fprintf(ft,"# input unpack                     --> mean: %.3f ms, stdev: %.3f ms\n", mean[2], stdev[2]);
      fprintf(ft,"# compute fep combinations         --> mean: %.3f ms, stdev: %.3f ms\n", mean[3], stdev[3]);
      fprintf(ft,"# fit data and set output          --> mean: %.3f ms, stdev: %.3f ms\n", mean[4], stdev[4]);
      fprintf(ft,"# copy output (DtoH)               --> mean: %.3f ms, stdev: %.3f ms\n", mean[5], stdev[5]);
    

      for (int j=0 ; j < (N_LOOPS); j++) {
        for (int t=0; t < 6; ++t)
          fprintf(ft,"%.3f ",times_array[t][j]);
        fprintf(ft,"\n");
      }
    } else { // CPU
      float mean[4];
      float stdev[4];
      for (int t=0; t < 4; ++t)
        get_mean(times_array_cpu[t], N_LOOPS, &mean[t], &stdev[t]);

      sprintf(fileTimes, "ListTimesCPU-Evts_%d_Loops_%d.txt", NEVTS, N_LOOPS);

      ft = fopen(fileTimes, "w");
      fprintf(ft,"# #NEvts: %d, Loops: %d, mean: %.3f ms, stdev: %.3f ms\n", NEVTS, N_LOOPS, mean[0], stdev[0]);
      fprintf(ft,"# input unpack                     --> mean: %.3f ms, stdev: %.3f ms\n", mean[1], stdev[1]);
      fprintf(ft,"# compute fep combinations         --> mean: %.3f ms, stdev: %.3f ms\n", mean[2], stdev[2]);
      fprintf(ft,"# fit data and set output          --> mean: %.3f ms, stdev: %.3f ms\n", mean[3], stdev[3]);

      for (int j=0 ; j < (N_LOOPS); j++) {
        for (int t=0; t < 4; ++t)
          fprintf(ft,"%.3f ",times_array_cpu[t][j]);
        fprintf(ft,"\n");
      }
    }

    fclose(ft);

    printf("All done. See %s for timing.\n", fileTimes);
  }


  if ( PRIORITY ) {
    if ( VERBOSE ) printf("Unlocking control...\n");
    unlock(semid);
  }

  free(data_send);
  free(tf);

  return 0;
}
