#include <unistd.h>
#include <getopt.h>
#include <sys/time.h>
#include "svt_utils.h"
#include <math.h>

#include <sched.h>
//#include "semaphore.c"
//#include <thrust/device_vector.h>

#include "svtsim_functions.h"


// global variables
int VERBOSE = 1;
int TIMER = 0;

#define __global__



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

#ifdef NOTDEF
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
#endif // NOTDEF
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


  while (n_iters--) {

      if ( TIMER ) gettimeofday(&time_start, NULL);
      
      if ( VERBOSE ) printf("Start working on CPU, iteration %d..... \n", n_iters);
      
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
      

    if ( TIMER ) {
      if ( n_iters < N_LOOPS ) { // skip the first "skip" iterations
        timerange = ((time_stop.tv_usec + 1000000 * time_stop.tv_sec) - (time_start.tv_usec + 1000000 * time_start.tv_sec))/1000.0;
	 //CPU
          times_array_cpu[0][n_iters] = timerange; 
          for (int t=1; t < 4; ++t)
            times_array_cpu[t][n_iters] = ptime_cpu[t-1];
        }
    }

  } // end iterations


  if ( TIMER ) {
    gettimeofday(&tEnd, NULL);
    timerange = ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0;
    if ( VERBOSE ) printf("Time to complete all: %.3f ms\n", timerange); 
  }

// write output file
FILE* OUTCHECK = fopen(fileOut, "w");
for (int i=0; i < tf->out->ndata; i++)
      fprintf(OUTCHECK,"%.6x\n", tf->out->data[i]);

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



  free(data_send);
  free(tf);

  return 0;
}
