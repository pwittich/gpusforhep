#include <unistd.h>
#include <sys/time.h>
#include "svt_utils.h"


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

void svt_GPU(tf_arrays_t tf, unsigned int *data_in, int n_words) {

  start_time();
  int tEvts=0;
  dim3 blocks(NEVTS,MAXROAD);

  // Cuda Malloc
  int* d_tEvts;
  MY_CUDA_CHECK(cudaMalloc((void**)&d_tEvts, sizeof(int)));
  struct evt_arrays* evt_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&evt_dev, sizeof(evt_arrays)));
  struct extra_data *edata_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&edata_dev, sizeof(extra_data)));
  struct fep_arrays *fep_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&fep_dev, sizeof(fep_arrays)));
  struct fit_arrays *fit_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&fit_dev, sizeof(fit_arrays)));
  struct fout_arrays *fout_dev;
  MY_CUDA_CHECK(cudaMalloc((void**)&fout_dev, sizeof(fout_arrays)));

  // input copy HtoD
  int len;
  len = SVTSIM_NBAR * FITBLOCK * sizeof(int);
  MY_CUDA_CHECK(cudaMemcpy(edata_dev->whichFit, tf->whichFit, len, cudaMemcpyHostToDevice));
  len = NFITPAR * (DIMSPA+1) * SVTSIM_NBAR * FITBLOCK * sizeof(long long int);
  MY_CUDA_CHECK(cudaMemcpy(edata_dev->lfitparfcon, tf->lfitparfcon, len, cudaMemcpyHostToDevice));
  len = NEVTS * sizeof(int);
  MY_CUDA_CHECK(cudaMemcpy(edata_dev->wedge, tf->wedge, len, cudaMemcpyHostToDevice));

  // initialize structures
  init_arrays_GPU<<<blocks, NSVX_PLANE+1>>>(fout_dev, evt_dev, d_tEvts);

  // Unpack
  gf_unpack_GPU(data_in, n_words, evt_dev, d_tEvts );
  stop_time("input unpack");

  MY_CUDA_CHECK(cudaMemcpy(&tEvts, d_tEvts, sizeof(int), cudaMemcpyDeviceToHost));
  tf->totEvts = tEvts;
 
  // Fep comb and set
  start_time();  
  gf_fep_GPU( evt_dev, fep_dev, tEvts );
  stop_time("compute fep combinations");
  
  // Fit and set Fout
  start_time();
  gf_fit_GPU(fep_dev, evt_dev, edata_dev, fit_dev, fout_dev, tEvts);
  stop_time("fit data and set output");

  // Output copy DtoH

  start_time();
  MY_CUDA_CHECK(cudaMemcpy(tf->fout_ntrks, fout_dev->fout_ntrks, NEVTS * sizeof(int), cudaMemcpyDeviceToHost));
  MY_CUDA_CHECK(cudaMemcpy(tf->fout_ee_word, fout_dev->fout_ee_word, NEVTS * sizeof(int), cudaMemcpyDeviceToHost));
  MY_CUDA_CHECK(cudaMemcpy(tf->fout_gfword, fout_dev->fout_gfword, NEVTS * MAXROAD * MAXCOMB * NTFWORDS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  stop_time("copy output (DtoH)");

  start_time();
  MY_CUDA_CHECK( cudaFree(evt_dev) );
  MY_CUDA_CHECK( cudaFree(fep_dev) );
  MY_CUDA_CHECK( cudaFree(edata_dev) );
  MY_CUDA_CHECK( cudaFree(fit_dev) );
  MY_CUDA_CHECK( cudaFree(fout_dev));
  MY_CUDA_CHECK( cudaFree(d_tEvts));
  stop_time("cudaFree structures");
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


  if ( strcmp(where,"gpu") == 0 ) {

    gettimeofday(&tBegin, NULL);
    // this is just to measure time to initialize GPU
    cudaEvent_t     init;
    MY_CUDA_CHECK( cudaEventCreate( &init ) );

    gettimeofday(&tEnd, NULL);

    printf("Time to initialize GPU: %.3f secs\n",
          ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000000.0);

  }

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

  tf_arrays_t tf;

  gf_init(&tf);
  svtsim_fconread(tf);

  gettimeofday(&tBegin, NULL);

  if ( strcmp(where,"cpu") == 0 ) { // CPU
    printf("Start work on CPU..... \n");
    
    gettimeofday(&ptBegin, NULL);
    gf_fep_unpack(tf, k, data_send);
    gettimeofday(&ptEnd, NULL);
    printf("Time to CPU unpack: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    gettimeofday(&ptBegin, NULL);
    gf_fep_comb(tf);
    gettimeofday(&ptEnd, NULL);
    printf("Time to CPU comb: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    gettimeofday(&ptBegin, NULL);
    gf_fit(tf);
    gf_comparator(tf);
    gettimeofday(&ptEnd, NULL);
    printf("Time to CPU fit: %.3f ms\n",
          ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0);

    printf(".... fits %d events! \n", tf->totEvts);

  } else { // GPU
    printf("Start work on GPU..... \n");
    svt_GPU(tf, data_send, k);
    printf(".... fits %d events! \n", tf->totEvts);
    // build "cable" output structure
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
