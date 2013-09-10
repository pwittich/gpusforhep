#include <unistd.h>
#include <cstring>
#include <sys/time.h>
#include <memory>
#include "mic.h"

int svt_mic(tf_arrays* tf, extra_data edata, unsigned int *data_in, int n_words, unsigned int *data_rec)
{
  int totEvts = 0;
  int ie = 0;
  int ow =0;
  evt_arrays* evt_dev = 0;
  fep_arrays* fep_dev = 0;
  fit_arrays* fit_dev = 0;
  fout_arrays* fout_dev = 0;

  struct timeval ptBegin, ptEnd;
  float ptime = 0;

  std::cout << "Start unpacking...\n" << std::flush;

  gettimeofday(&ptBegin, NULL);

#pragma offload target(mic) nocopy(evt_dev) nocopy(fep_dev) nocopy(fit_dev) nocopy(fout_dev)
  mic::init(evt_dev, fep_dev, fit_dev, fout_dev);

  gettimeofday(&ptEnd, NULL);
  ptime = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
  printf("Time to init: %.3f ms\n", ptime);


  gettimeofday(&ptBegin, NULL);
#pragma offload target(mic) in(data_in:length(n_words)) nocopy(evt_dev) out(totEvts)
  mic::gf_unpack(data_in, n_words, evt_dev, totEvts);

  gettimeofday(&ptEnd, NULL);
  ptime = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
  printf("Time to unpack: %.3f ms\n", ptime);

  std::cout << "Start fep...\n" << std::flush;

  gettimeofday(&ptBegin, NULL);

#pragma offload target(mic) nocopy(evt_dev) nocopy(fep_dev)
{
  #pragma omp parallel for
  for (ie=0; ie < totEvts; ie++) {
    mic::gf_fep_comb_Mic(evt_dev, fep_dev, ie);
  }
}

  gettimeofday(&ptEnd, NULL);
  ptime = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
  printf("Time to fep: %.3f ms\n", ptime);

  std::cout << "Start fit...\n" << std::flush;
/*
  gettimeofday(&ptBegin, NULL);

#pragma offload target(mic) nocopy(fit_dev) nocopy(fout_dev)
  mic::init_fout(fout_dev, fit_dev);

  gettimeofday(&ptEnd, NULL);
  ptime = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
  printf("Time to init: %.3f ms\n", ptime);
*/
  gettimeofday(&ptBegin, NULL);

#pragma offload target(mic) in(edata) nocopy(fep_dev) nocopy(fit_dev) nocopy(evt_dev) nocopy(fout_dev)
{
  #pragma omp parallel for
  for (ie=0; ie < totEvts; ie++) {
    mic::kFit(fep_dev, &edata, fit_dev, ie);
    mic::gf_fit_format_Mic(fep_dev, fit_dev, ie);
    mic::gf_comparator_Mic(fep_dev, evt_dev, fit_dev, fout_dev, ie);
  }
}

#pragma offload target(mic) nocopy(fout_dev) in(totEvts) out(data_rec:length(n_words) free_if(0)) out(ow)
  mic::set_outcable(fout_dev, totEvts, data_rec, ow);

#pragma offload target(mic) nocopy(evt_dev) nocopy(fep_dev) nocopy(fit_dev) nocopy(fout_dev)
  mic::destroy(evt_dev, fep_dev, fit_dev, fout_dev);

  gettimeofday(&ptEnd, NULL);
  ptime = ((ptEnd.tv_usec + 1000000 * ptEnd.tv_sec) - (ptBegin.tv_usec + 1000000 * ptBegin.tv_sec))/1000.0;
  printf("Time to fit: %.3f ms\n", ptime);

  std::cout << " done " << totEvts << " events\n" << std::flush;

  tf->totEvts = totEvts;

  return ow;
}

void help(char* prog) {

  printf("Use %s [-i fileIn] [-o fileOut] [-s cpu || gpu] [-h] \n\n", prog);
  printf("  -i fileIn       Input file (Default: hbout_w6_100evts).\n");
  printf("  -o fileOut      Output file (Default: gfout.txt).\n");
  printf("  -h              This help.\n");

}


int main(int argc, char* argv[]) {

  int c;
  char const* fileIn = "hbout_w6_100evts";
  char const* fileOut = "gfout.txt";

  while ( (c = getopt(argc, argv, "i:s:o:h")) != -1 ) {
    switch(c) {
      case 'i': 
        fileIn = optarg;
	      break;
      case 'o':
        fileOut = optarg;
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
  int k=0;
  while (fscanf(hbout, "%s", word) != EOF) {
    hexaval = strtol(word,NULL,16);
    data_send[k] = hexaval;
    k++;
  }

  fclose(hbout);

  tf_arrays_t tf;
  extra_data *edata = (extra_data*)malloc(sizeof(struct extra_data));

  gf_init(&tf);
  svtsim_fconread(tf, edata);

  std::cout << "Initialise Mic...\n" << std::flush;

  gettimeofday(&tBegin, NULL);
#pragma offload target(mic)
  mic::empty();

  gettimeofday(&tEnd, NULL);
  float ptime = ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0;
  printf("Time to start mic: %.3f ms\n", ptime);

  gettimeofday(&tBegin, NULL);

  printf("Start work on MIC...\n");
  outword = svt_mic(tf, *edata, data_send, k, data_rec);
  printf(".... fits %d events! \n", tf->totEvts);

  gettimeofday(&tEnd, NULL);
  printf("Time to complete all: %.3f ms\n",
          ((tEnd.tv_usec + 1000000 * tEnd.tv_sec) - (tBegin.tv_usec + 1000000 * tBegin.tv_sec))/1000.0);

  // write output file
  FILE* OUTCHECK = fopen(fileOut, "w");

  for (int i=0; i < outword ; i++)
    fprintf(OUTCHECK,"%.6x\n", data_rec[i]);

  fclose(OUTCHECK);

  free(data_send);
  free(data_rec);
  free(edata);  
  free(tf);

  return 0;
}
