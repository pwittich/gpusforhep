#include <unistd.h>
#include <cstring>
#include <sys/time.h>
#include <memory>
#include <math.h>
#include "mic.h"

// calculate mean and stdev on an array of count floats
void get_mean(float* times_array, int count, float *mean, float *stdev) {

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

// returns the number of output words
int svt_mic(extra_data edata, unsigned int *data_in, int n_words, unsigned int *data_rec, int* tev, float timer[3])
{
  int totEvts = 0;
  int ie = 0;
  int ow = 0;
  evt_arrays* evt_dev = 0;
  fep_arrays* fep_dev = 0;
  fit_arrays* fit_dev = 0;
  fout_arrays* fout_dev = 0;

  struct timeval ptBegin, ptEnd;

#pragma offload target(mic) \
  in(data_in:length(n_words)) in(edata) \
  inout(totEvts)  \
  nocopy(evt_dev) nocopy(fep_dev) nocopy(fit_dev) nocopy(fout_dev) \
  out(data_rec:length(n_words)) out(ow) out(timer:length(3))
{

  // Unpack
  gettimeofday(&ptBegin, NULL);


// without thrust
  long sizeW = sizeof(int) * n_words;
  int *ids  = (int *)malloc(sizeW);
  int *out1 = (int *)malloc(sizeW);
  int *out2 = (int *)malloc(sizeW);
  int *out3 = (int *)malloc(sizeW);

  #pragma omp parallel for
  for (int idx=0; idx < n_words; idx++) {
    mic::k_word_decode(idx, data_in, ids, out1, out2, out3);
  }
  mic::init_evt(evt_dev, fep_dev);
  mic::gf_unpack_wot(n_words, ids, out1, out2, out3, evt_dev, totEvts);

  free(ids);
  free(out1);
  free(out2);
  free(out3);

// thrust version
/*
  mic::init_evt(evt_dev, fep_dev);
  mic::gf_unpack(data_in, n_words, evt_dev, totEvts);
*/

  gettimeofday(&ptEnd, NULL);
  timer[0] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

  // Fep
  gettimeofday(&ptBegin, NULL);

  #pragma omp parallel for
  for (ie=0; ie < totEvts; ie++) {
    mic::init_fep(fep_dev, ie);
    mic::gf_fep_comb_Mic(evt_dev, fep_dev, ie);
  }

  gettimeofday(&ptEnd, NULL);
  timer[1] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

  // Fit/Fout
  gettimeofday(&ptBegin, NULL);

  mic::init_fit(fit_dev, fout_dev);
  #pragma omp parallel for
  for (ie=0; ie < totEvts; ie++) {
    mic::kFit(fep_dev, &edata, fit_dev, ie);
    mic::gf_fit_format_Mic(fep_dev, fit_dev, ie);
    mic::gf_comparator_Mic(fep_dev, evt_dev, fit_dev, fout_dev, ie);
  }
  mic::set_outcable(fout_dev, totEvts, data_rec, ow);
  mic::destroy(evt_dev, fep_dev, fit_dev, fout_dev);

  gettimeofday(&ptEnd, NULL);
  timer[2] = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;

} // end offload

  *tev = totEvts;
  return ow;

}

void help(char* prog) {

  printf("Use %s [-i fileIn] [-o fileOut] [-l #loops] [-h] \n\n", prog);
  printf("  -i fileIn       Input file (Default: hbout_w6_100evts).\n");
  printf("  -o fileOut      Output file (Default: gfout.txt).\n");
  printf("  -l loops        Number of executions (Default: 1).\n");
  printf("  -h              This help.\n");

}


int main(int argc, char* argv[]) {

  int c;
  char const* fileIn = "hbout_w6_100evts";
  char const* fileOut = "gfout.txt";
  int N_LOOPS = 1;

  while ( (c = getopt(argc, argv, "i:l:o:h")) != -1 ) {
    switch(c) {
      case 'i': 
        fileIn = optarg;
	      break;
      case 'o':
        fileOut = optarg;
        break;
      case 'l':
        N_LOOPS = atoi(optarg);
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

  // read input file
  printf("Opening file %s\n", fileIn);
  FILE* hbout = fopen(fileIn,"r");

  if(hbout == NULL) {
    printf("Cannot open input file\n");
    exit(1);
  }

  unsigned int hexaval;
  int numword = 2500000;
  int outword = 0;
  unsigned int *data_send = (unsigned int*)malloc(numword*sizeof(unsigned));
  unsigned int *data_rec = (unsigned int*)malloc(numword*sizeof(unsigned));
  if ( data_send == (unsigned int*) NULL ) {
    perror("malloc");
    return 2;
  }
  
  char word[16];
  numword = 0;
  while (fscanf(hbout, "%s", word) != EOF) {
    hexaval = strtol(word,NULL,16);
    data_send[numword] = hexaval;
    numword++;
  }

  fclose(hbout);

  tf_arrays_t tf;
  extra_data *edata = (extra_data*)malloc(sizeof(struct extra_data));

  gf_init(&tf);
  svtsim_fconread(tf, edata);

  free(tf);

  gettimeofday(&tBegin, NULL);

#pragma offload target(mic)
  mic::empty();

  gettimeofday(&tEnd, NULL);
  float initmic = ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0;
  printf("Time to start mic: %.3f ms\n", initmic);

  printf("Start work on MIC...\n");
 
  // Do we want to skip the first "skip" runs from mean calculation?
  int skip = 5;
  int n_iters = N_LOOPS+skip;
  float ptime[3];
  float times[4][N_LOOPS];
  int totEvts;

  while (n_iters--) {

    totEvts = 0;
    gettimeofday(&tBegin, NULL);
    outword = svt_mic(*edata, data_send, numword, data_rec, &totEvts, ptime);
    gettimeofday(&tEnd, NULL);
    if ( n_iters < N_LOOPS ) { // skip the first "skip" iterations
      times[0][n_iters] = ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0;
      for (int i = 0; i < 3; i++)
        times[i+1][n_iters] = ptime[i];
//    printf("Time to fits %d events: %.3f ms\n", totEvts, times[0][n_iters]);
    }

  }

  // write output file
  FILE* OUTCHECK = fopen(fileOut, "w");

  for (int i=0; i < outword ; i++)
    fprintf(OUTCHECK,"%.6x\n", data_rec[i]);

  fclose(OUTCHECK);

  // write file with times
  char fileTimes[1024];
  FILE *ft;
  float mean[4];
  float stdev[4];
  for (int t=0; t < 4; ++t)
    get_mean(times[t], N_LOOPS, &mean[t], &stdev[t]);

  sprintf(fileTimes, "ListTimesMIC-Evts_%d_Loops_%d.txt", NEVTS, N_LOOPS);

  ft = fopen(fileTimes, "w");
  fprintf(ft,"# #NEvts: %d, Loops: %d, mean: %.3f ms, stdev: %.3f ms\n", totEvts, N_LOOPS, mean[0], stdev[0]);
  fprintf(ft,"# Time to initialize MIC: %.3f ms.\n", initmic);
  fprintf(ft,"# step 1: input unpack                     --> mean: %.3f ms, stdev: %.3f ms\n", mean[1], stdev[1]);
  fprintf(ft,"# step 2: compute fep combinations         --> mean: %.3f ms, stdev: %.3f ms\n", mean[2], stdev[2]);
  fprintf(ft,"# step 3: fit data and set output          --> mean: %.3f ms, stdev: %.3f ms\n", mean[3], stdev[3]);

  fprintf(ft,"# All\t\t\t st1\t\t st2\t\t st3\n");
  for (int j=0 ; j < (N_LOOPS); j++) {
    for (int t=0; t < 4; ++t)
      fprintf(ft,"%.3f\t\t",times[t][j]);
    fprintf(ft,"\n");
  }
 
  fclose(ft);

  printf("All done. See %s for timing.\n", fileTimes);

  free(data_send);
  free(data_rec);
  free(edata);  

  return 0;
}
