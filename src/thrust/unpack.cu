// 
// $Id: unpack.cu 69 2013-03-11 20:45:13Z poprocki $
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
#include <cassert>
#include <asm/types.h>
using namespace std;

// typedefs for shorthand
typedef thrust::tuple<unsigned int, unsigned int, unsigned int> IntTuple;
typedef thrust::device_vector<unsigned int>::iterator           IntIterator;
typedef thrust::tuple<IntIterator, IntIterator, IntIterator>    IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple>                     ZipIterator;


//#define TIMING_DEETS

// copied from cutil.h
#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }


// which buffer to unpack to, based on input value
struct unpackBuffer { // : public thrust::unary_function<unsigned int, unsigned int> {
  unsigned int val_;
  unpackBuffer(int vval) : val_(vval) {} // constructor
  __host__ __device__ 
  int operator()(unsigned int i ) {
    if ( (i&0x3U) == val_ ) 
      return 1;
    else
      return 0;
  }
  __host__ __device__ 
  int operator()(IntTuple tup ) {
    if ( (thrust::get<0>(tup) & 0x3U) == val_ ) 
      return 1;
    else
      return 0;
  }
};

// unpacking function itself. unpack the passed-in int to the tuple.
struct unpacker : public thrust::unary_function<unsigned int, IntTuple> {

    // pointers to three output buffers
  __host__ __device__ 
  IntTuple operator()(unsigned int i ) {
    unsigned int val1, val2, val3;
    val1 = i&0x3U;
    val2 = i&(0xffU<<8);
    val3 = i&(0xffU<<16);
    return thrust::make_tuple(val1,val2,val3);
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
    CUDA_SAFE_CALL(cudaSetDevice(max_device));
    printf("# Running on device %d (name %s)\n", max_device, best_prop.name);
  }

  // input data
  thrust::host_vector<unsigned int> h_vec(nwords);
  printf("h_vec size is %u\n", h_vec.size());


  // these three vectors are of fixed size in this code.
  thrust::device_vector<unsigned int> d_map(nwords,0U);
  thrust::device_vector<unsigned int> d_vec(nwords,0U);

  thrust::device_vector<unsigned int> d_out1(nwords,0U);
  thrust::device_vector<unsigned int> d_out2(nwords,0U);
  thrust::device_vector<unsigned int> d_out3(nwords,0U);

  // debugging
  unsigned int *h_out1, *h_out2, *h_out3;
  h_out1 = (unsigned int*)malloc(sizeof(unsigned int)*nwords);
  h_out2 = (unsigned int*)malloc(sizeof(unsigned int)*nwords);
  h_out3 = (unsigned int*)malloc(sizeof(unsigned int)*nwords);
  assert(!( (h_out1 == NULL ) || (h_out2 == NULL) || (h_out3 == NULL ) ));

  //float tbar = 0.0, tsqbar = 0.0;
  //int n = 0;
  int ndiffs = 0;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // GPU RUNNING STARTS HERE
  cudaEventRecord(start, 0);
  for ( int ev = 0; ev < ntries && !saw_sigint; ++ev ) {
    if ( ev % 50 == 0 ) {
      fprintf(stderr, "Step %i\n", ev);
    }
    // generate some random numbers serially, on host
    std::generate(h_vec.begin(), h_vec.end(), rand);

    // transfer data to the device
    d_vec = h_vec;

    // where to copy to test
    //thrust::transform(d_vec.begin(), d_vec.end(), d_map.begin(), unpackBuffer(3));

    // three copy_if's
    // thrust::zip_iterator<thrust::tuple<thrust::device_vector<unsigned int>::iterator, 
    // 				       thrust::device_vector<unsigned int>::iterator, 
    // 				       thrust::device_vector<unsigned int>::iterator > > end1 =
    //   thrust::copy(
    // 		   thrust::make_transform_iterator(d_vec.begin(), unpacker()),
    // 		   thrust::make_transform_iterator(d_vec.end(), unpacker()),
    // 		   thrust::make_zip_iterator(thrust::make_tuple(d_out1.begin(),
    // 								d_out2.begin(),
    // 								d_out3.begin())));

    ZipIterator zipOut = thrust::make_zip_iterator(thrust::make_tuple(d_out1.begin(),
								      d_out2.begin(),
								      d_out3.begin()));
    ZipIterator end1 = thrust::copy_if(
	thrust::make_transform_iterator(d_vec.begin(), unpacker()),
	thrust::make_transform_iterator(d_vec.end(), unpacker()),
	zipOut,
	unpackBuffer(3U));

    size_t out1 = end1 - zipOut;

    //
    thrust::host_vector<unsigned int> h_test1(nwords), h_test2(nwords), h_test3(nwords);
    //1thrust::host_vector<unsigned int> h_test4(nwords), h_test5(nwords,0);
    assert(out1<nwords);
    thrust::copy_n(d_out1.begin(), out1, h_test1.begin());
    thrust::copy_n(d_out2.begin(), out1, h_test2.begin());
    thrust::copy_n(d_out3.begin(), out1, h_test3.begin());
    // thrust::copy_n(d_vec.begin(), nwords, h_test4.begin());
    // thrust::copy_n(d_map.begin(), nwords, h_test5.begin());
    // computer
    int nout0 = 0;
    for ( int i = 0; i < nwords; ++i ) {
      //printf("%u: %08x\t%08x\t%u", i, h_vec[i], h_test4[i], h_test5[i]);
      if ( (h_vec[i] & 0x3U) == 0x3U ) {
	//printf("--> pass");
	h_out1[nout0] = h_vec[i] & 0x3u;
	h_out2[nout0] = h_vec[i] & (0xffu<<8);
	h_out3[nout0] = h_vec[i] & (0xffu<<16);
	++nout0;
      }
      //printf("\n");
    }
    if ( nout0 != out1 ) {
      printf("ERROR: nout0 = %u, out1 = %u\n", nout0, out1);
    }
    if ( nout0 != out1 ) {
      for ( int i =0; i < max(nout0, out1); ++i ) {
	printf("%d:\t0x%08x\t0x%08x\t0x%08x\t0x%08x\t0x%08x\t0x%08x\n",i,
	       h_out1[i], h_out2[i], h_out3[i],
	       h_test1[i], h_test2[i], h_test3[i]);
      }
    }
    for ( int i = 0; i < nout0; ++i ) {
      bool mism = false;
      if ( h_out1[i] != h_test1[i] ) {
	if ( ! mism ) printf("%d: ", i);
	printf("mism: out1 0x%06x 0x%06x, ", h_out1[i], h_test1[i]);
	mism = true;
      }
      if ( h_out2[i] != h_test2[i] ) {
	if ( ! mism ) printf("%d: ", i);
	printf("mism: out2 0x%06x 0x%06x, ", h_out2[i], h_test2[i]);
	mism = true;
      }
      if ( h_out3[i] != h_test3[i] ) {
	if ( ! mism ) printf("%d: ", i);
	printf("mism: out3 0x%06x 0x%06x, ", h_out3[i], h_test3[i]);
	mism = true;
      }
      if ( mism ) {
	++ndiffs;
	printf("\n");
      }
    }
  } // loop over events

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= (float)ntries;
  printf("cuda time: %5.3f ms\n", ms);
  
  printf("ndiffs = %d\n", ndiffs);
  return 0;
}
