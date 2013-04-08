// 
// $Id: markpack.cu 66 2013-03-01 22:14:23Z wittich $
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include "slinktest/src/NodeUtils.hh"

//#include <algorithm>
#include <cstdlib>
#include <asm/types.h>
using namespace std;

//#define TIMING_DEETS

// copied from cutil.h
#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }


  
struct testChi2: public thrust::unary_function<unsigned int, int> {
  __host__ __device__ 
  int operator()(unsigned int i ) {
    unsigned int chi2 = (i&0xFFU);
    if ( chi2 < 16 ) 
      return 1;
    return 0;
  }
};

#include <signal.h>
static int saw_sigint = 0;
void siginthandler(int signal)
{
  saw_sigint = 1;
  return;
}


int main(int argc, char **argv)
{
  char *progname;
 
  progname = strrchr(argv[0], '/');
  if ( progname == (char *) NULL )
    progname = argv[0];
  else
    progname++;
  extern char *optarg;
  int ntries = 1000;
  int shift = 16;
  char o;
  while (( o = getopt (argc, argv, "dSr:i:s:n:o:hb:e:")) != EOF) {
    switch (o) {
    case 'n':
      ntries = atoi(optarg);
      break;
    case 's':
      shift = atoi(optarg);
      break;
    }
  }
  int nwords = (1<<shift);
  fprintf(stderr, "Running over %d trials of %d words\n", ntries, nwords);
  fprintf(stdout, "# Running over %d trials of %d words\n", ntries, nwords);

  signal(SIGINT, siginthandler); 

  int num_devices, device;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices));
  if ( num_devices > 1 ) {
    int max_multiprocessors = 0, max_device = 0;
    cudaDeviceProp best_prop;
    for ( device = 0; device < num_devices; ++device ) {
      cudaDeviceProp properties;
      CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
      if ( max_multiprocessors < properties.multiProcessorCount ) {
	max_multiprocessors = properties.multiProcessorCount;
	max_device = device;
	best_prop = properties;
      }
    }
    cudaSetDevice(max_device);
    printf("# Running on device %d (name %s)\n", max_device, best_prop.name);
  }


  thrust::host_vector<unsigned int> h_vec(nwords);

  // map
  // result of chi2 test
  // these three vectors are of fixed size in this code.
  thrust::device_vector<unsigned int> d_chi2(nwords);
  thrust::device_vector<unsigned int> d_map(nwords);
  thrust::device_vector<unsigned int> d_vec(nwords);

  float tbar = 0.0, tsqbar = 0.0;
  int n = 0;
  int ndiffs = 0;
  // GPU RUNNING STARTS HERE
  for ( int ev = 0; ev < ntries && !saw_sigint; ++ev ) {
    if ( ev % 50 == 0 ) {
      fprintf(stderr, "Step %i\n", ev);
    }

    __u32 t[11];
    memset(&t[0],0, sizeof(__u32)*11);
    //printf("%u %u %u\n", t[4], t[5], t[6]);
    rdtscl(t[0]);
    // generate some random numbers serially, on host
    std::generate(h_vec.begin(), h_vec.end(), rand);
    rdtscl(t[1]);
    // transfer data to the device
    d_vec = h_vec;
#ifdef TIMING_DEETS    
    cudaDeviceSynchronize(); // block until kernel is finished
#endif // TIMING_DEETS
    rdtscl(t[2]);

    // chi2 test
    thrust::transform(d_vec.begin(), d_vec.end(), d_chi2.begin(), testChi2());
#ifdef TIMING_DEETS    
    cudaDeviceSynchronize(); // block until kernel is finished
#endif // TIMING_DEETS
    rdtscl(t[3]);
    //int nout = end - d_chi2.begin();
    //int nout = thrust::count(d_chi2.begin(), d_chi2.end(), 1);
    //int nout = thrust::count_if(d_chi2.begin(), d_chi2.end(), thrust::placeholders::_1 == 1);
    
#ifdef TIMING_DEETS    
    cudaDeviceSynchronize(); // block until kernel is finished
#endif // TIMING_DEETS
    rdtscl(t[4]);
    //printf("nout = %d\n", nout);
    thrust::device_vector<unsigned int> d_output(nwords);
#ifdef TIMING_DEETS    
    cudaDeviceSynchronize(); // block until kernel is finished
#endif // TIMING_DEETS
    rdtscl(t[5]);

    // copy to output start
    thrust::device_vector<unsigned int>::iterator end = 
      thrust::copy_if(d_vec.begin(), d_vec.end(), d_output.begin(), testChi2());
#ifdef TIMING_DEETS    
    cudaDeviceSynchronize(); // block until kernel is finished
#endif // TIMING_DEETS
    rdtscl(t[6]);
    int nout = end - d_output.begin();

    // transfer data back to host
    thrust::host_vector<unsigned int> h_out(nout);
#ifdef TIMING_DEETS    
    cudaDeviceSynchronize(); // block until kernel is finished
#endif // TIMING_DEETS
    rdtscl(t[7]);
    thrust::copy_n(d_output.begin(), nout, h_out.begin());
#ifdef TIMING_DEETS    
    cudaDeviceSynchronize(); // block until kernel is finished
#endif // TIMING_DEETS
    rdtscl(t[8]);
    // copy to output end


    // ----------------------------------------
    // DEBUG
    // ----------------------------------------

    // test data
    // thrust::host_vector<unsigned int> h_test(nwords), h_test1(nwords), h_test2(nwords);
    // thrust::copy(d_map.begin(), d_map.end(), h_test.begin());
    // thrust::copy(d_vec.begin(), d_vec.end(), h_test1.begin());
    // thrust::copy(d_chi2.begin(), d_chi2.end(), h_test2.begin());
    //printf("Event = %d\n", ev);
    // for ( int i = 0; i < 120; ++i ) {
    //   printf("map output %u\t%u\t%08x\t%08x\n", i, h_test[i], h_test1[i], h_test2[i]);
    // }


    /// END GPU
    // fprintf(stderr, "outsize is  %i (%5.2f)\n",  nout, (1.0*nout/h_vec.size()));
    float time_us = tstamp_to_us(t[1], t[8]);  
    for ( int i = 0; i < 8; ++i ) {
      float dt = tstamp_to_us(t[i], t[i+1]);  
      fprintf(stdout, "%5.2f ", dt);
    }
    fprintf(stdout, "%5.2f ", time_us);
    //fprintf(stdout, "\n");

    tsqbar += time_us * time_us;
    tbar += time_us;
    ++n;

    // Repeat on CPU
    rdtscl(t[9]);
    std::vector<unsigned int> output;
    for (int i = 0; i < h_vec.size(); ++i ) {
      unsigned int chi2 = (h_vec[i] & 0xFFU);

      if ( chi2 < 16 ) {
	output.push_back(h_vec[i]);
      }
    }
    rdtscl(t[10]);
    time_us = tstamp_to_us(t[9], t[10]);  
    fprintf(stdout, "%5.2f ", time_us);
    fprintf(stdout, "\n");



    int mismatches = 0;
    if ( nout != output.size() ) {
      fprintf(stderr, "different data sizes: device = %d, host = %d\n", nout, output.size());
    }
    for ( int i = 0; i < output.size(); ++i ) {
    //for ( int i = 0; i < 10; ++i ) {
      //fprintf(stderr, "%d\t%08x\t%08x", i, h_out[i], output[i]);
      if ( h_out[i] != output[i] ) {
	fprintf(stderr, "\t\t-->mismatch");
	// // is the wrong data on the device
	// for (int j = 0; j < nwords ; ++j ) {
	//   if (h_out[i] == h_vec[j] ) {
	//     fprintf(stderr, " correct copy: %08x (%d)", h_test1[j], j); 
	//   }
	// }
	// // is the right data on the device
	// for (int j = 0; j < nwords ; ++j ) {
	//   if (output[i] == h_test1[j] ) {
	//     fprintf(stderr, " real exists: %08x (%d)", h_test1[j], j);
	//   }
	// }
	++mismatches;
      }
      //fprintf(stderr,"\n");
    }
    
    // fprintf(stderr, "n_out (device) = %d, n_out(host) = %d, mism = %d\n", nout, output.size(), 
    // 	    mismatches);
    ndiffs += mismatches;

  }
  float trms = sqrt(tsqbar - (tbar*tbar)/(1.0*n))/(n-1.0);
  tbar = tbar/(1.0*n);
  printf("# timing: %5.2f +- %5.2f us\n", tbar, trms);
  printf("# ndiffs = %d\n", ndiffs);

  return 0;
}
