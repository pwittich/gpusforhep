// -*-c++-*-
//#define LOCAL 1

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>


#include "slinktest/src/NodeUtils.hh"
//#include <sys/time.h>

#include <algorithm>
#include <cstdlib>
#include <asm/types.h>

using namespace std;

// copied from cutil.h
#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }
/*
#define cells_per_layer 2
#define random_init 4 // is 2^cells_per_layer
#define nCombinations 32 // is cells_per_layer^5
#define div_layer2 2 // is cells_per_layer^1
#define div_layer3 4 // is cells_per_layer^2
#define div_layer4 8 // is cells_per_layer^3
#define div_layer5 16 // is cells_per_layer^4
*/
/*
#define cells_per_layer 3
#define random_init 8 // is 2^cells_per_layer
#define nCombinations 243 // is cells_per_layer^5
#define div_layer2 3 // is cells_per_layer^1
#define div_layer3 9 // is cells_per_layer^2
#define div_layer4 27 // is cells_per_layer^3
#define div_layer5 81 // is cells_per_layer^4
*/
/*
#define cells_per_layer 4
#define random_init 16 // is 2^cells_per_layer
#define nCombinations 1024 // is cells_per_layer^5
#define div_layer2 4 // is cells_per_layer^1
#define div_layer3 16 // is cells_per_layer^2
#define div_layer4 64 // is cells_per_layer^3
#define div_layer5 256 // is cells_per_layer^4
*/
/*
#define cells_per_layer 5
#define random_init 32 // is 2^cells_per_layer
#define nCombinations 3125 // is cells_per_layer^5
#define div_layer2 5 // is cells_per_layer^1
#define div_layer3 25 // is cells_per_layer^2
#define div_layer4 125 // is cells_per_layer^3
#define div_layer5 625 // is cells_per_layer^4
*/
/*
#define cells_per_layer 6
#define random_init 64 // is 2^cells_per_layer
#define nCombinations 7776 // is cells_per_layer^5
#define div_layer2 6 // is cells_per_layer^1
#define div_layer3 36 // is cells_per_layer^2
#define div_layer4 216 // is cells_per_layer^3
#define div_layer5 1296 // is cells_per_layer^4
*/
/*
#define cells_per_layer 7
#define random_init 128 // is 2^cells_per_layer
#define nCombinations 16807 // is cells_per_layer^5
#define div_layer2 7 // is cells_per_layer^1
#define div_layer3 49 // is cells_per_layer^2
#define div_layer4 343 // is cells_per_layer^3
#define div_layer5 2401 // is cells_per_layer^4
*/

#define cells_per_layer 8
#define random_init 256 // is 2^cells_per_layer
#define nCombinations 32768 // is cells_per_layer^5
#define div_layer2 8 // is cells_per_layer^1
#define div_layer3 64 // is cells_per_layer^2
#define div_layer4 512 // is cells_per_layer^3
#define div_layer5 4096 // is cells_per_layer^4


struct check_hits
{
  template <typename Tuple>
  __host__ __device__ 
  void operator()( Tuple t ) {

    unsigned int index = thrust::get<0>(t);

    thrust::get<6>(t) = 
      ( (thrust::get<1>(t) & (1 << (index%cells_per_layer) )) >> ( index%cells_per_layer) ) + 
      ( (thrust::get<2>(t) & (1 << ((index/div_layer2)%cells_per_layer) )) >> ( (index/div_layer2)%cells_per_layer) ) + 
      ( (thrust::get<3>(t) & (1 << ((index/div_layer3)%cells_per_layer) )) >> ( (index/div_layer3)%cells_per_layer) ) + 
      ( (thrust::get<4>(t) & (1 << ((index/div_layer4)%cells_per_layer) )) >> ( (index/div_layer4)%cells_per_layer) ) + 
      ( (thrust::get<5>(t) & (1 << ((index/div_layer5)%cells_per_layer) )) >> ( (index/div_layer5)%cells_per_layer) ); 

  }
};

#include <signal.h>
static int saw_sigint = 0;
void siginthandler(int signal)
{
  saw_sigint = 1;
  return;
}

//float tstamp_to_us(struct timeval t1, struct timeval t2){
//  return (float)(t2.tv_sec*1e6 + t2.tv_usec - t1.tv_sec*1e6 - t1.tv_usec);
//}

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

  fprintf(stderr, "Running over %d trials\n", ntries);
  fprintf(stdout, "# Running over %d trials\n", ntries);

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

  srand(time(NULL));
  
  thrust::counting_iterator<unsigned int> index_first(0);
  thrust::counting_iterator<unsigned int> index_last( index_first + nCombinations );

  thrust::device_vector<uint8_t> combination(nCombinations);
  thrust::host_vector<uint8_t> combination_host(nCombinations);
  thrust::host_vector<uint8_t> combination_CPU(nCombinations);

  const bool print=false;

  float tbar = 0.0, tsqbar = 0.0;
  float tbar_cpu = 0.0, tsqbar_cpu = 0.0;
  int n = 0;
  //int ndiffs = 0;
  // GPU RUNNING STARTS HERE
  for ( int ev = 0; ev < ntries && !saw_sigint; ++ev ) {
    if ( ev % 50 == 0 ) {
      fprintf(stderr, "Step %i\n", ev);
    }

    uint8_t rand1 = std::rand()%random_init;
    uint8_t rand2 = std::rand()%random_init;
    uint8_t rand3 = std::rand()%random_init;
    uint8_t rand4 = std::rand()%random_init;
    uint8_t rand5 = std::rand()%random_init;
    
    if(print){

    for(int i=0; i<cells_per_layer; i++){
      if (i==0) printf("Layer 1: ");
      printf("%d", (rand1 & (1 << i))>>i );
      if(i==cells_per_layer-1) printf("\n");
    }
    for(int i=0; i<cells_per_layer; i++){
      if (i==0) printf("Layer 2: ");
      printf("%d", (rand2 & (1 << i))>>i );
      if(i==cells_per_layer-1) printf("\n");
    }
    for(int i=0; i<cells_per_layer; i++){
      if (i==0) printf("Layer 3: ");
      printf("%d", (rand3 & (1 << i))>>i );
      if(i==cells_per_layer-1) printf("\n");
    }
    for(int i=0; i<cells_per_layer; i++){
      if (i==0) printf("Layer 4: ");
      printf("%d", (rand4 & (1 << i))>>i );
      if(i==cells_per_layer-1) printf("\n");
    }
    for(int i=0; i<cells_per_layer; i++){
      if (i==0) printf("Layer 5: ");
      printf("%d", (rand5 & (1 << i))>>i );
      if(i==cells_per_layer-1) printf("\n");
    }

    }

    //struct timeval t[11];
    __u32 t[11];
    memset(&t[0],0, sizeof(__u32)*11);

    //gettimeofday(&t[0],NULL);
    rdtscl(t[0]);

    //setup the input hits
    thrust::constant_iterator<uint8_t> layer1( rand1 );
    thrust::constant_iterator<uint8_t> layer2( rand2 );
    thrust::constant_iterator<uint8_t> layer3( rand3 );
    thrust::constant_iterator<uint8_t> layer4( rand4 );
    thrust::constant_iterator<uint8_t> layer5( rand5 );
    
    //gettimeofday(&t[1],NULL);
    rdtscl(t[1]);

    //call the combination maker
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(index_first,layer1,layer2,layer3,layer4,layer5,combination.begin())),
		     thrust::make_zip_iterator(thrust::make_tuple(index_last,layer1,layer2,layer3,layer4,layer5,combination.end())),
		     check_hits());
    cudaDeviceSynchronize(); // block until kernel is finished

    //gettimeofday(&t[2],NULL);
    rdtscl(t[2]);

    //copy over the output
    combination_host = combination;
    cudaDeviceSynchronize(); // block until kernel is finished

    //gettimeofday(&t[3],NULL);
    rdtscl(t[3]);

    int hits_per_combination[] = {0,0,0,0,0,0};
    for(int i=0; i<nCombinations; i++){
      hits_per_combination[combination_host[i]]++;
	//printf("Combination %d had %d hits\n",i+1,combination_host[i]);
    }

    for(int i=0; i<6; i++){
      if(print) printf("%d combinations with %d hits.\n",hits_per_combination[i],i);
    }
    
    float time_us = tstamp_to_us(t[0], t[3]);  
    for ( int i = 0; i < 3; ++i ) {
      float dt = tstamp_to_us(t[i], t[i+1]);  
      if(print) fprintf(stdout, "%5.2f ", dt);
    }
    if(print) fprintf(stdout, "%5.2f ", time_us);
    if(print) fprintf(stdout, "\n");

    tsqbar += time_us * time_us;
    tbar += time_us;
    
    ++n;
    
    //check output

    // Repeat on CPU
    //gettimeofday(&t[4],NULL);
    rdtscl(t[4]);
    for(int i=0; i<nCombinations; i++){
      combination_CPU[i] = 
	( (rand1 & (1 << (i%cells_per_layer) )) >> ( i%cells_per_layer) ) + 
	( (rand2 & (1 << ((i/div_layer2)%cells_per_layer) )) >> ( (i/div_layer2)%cells_per_layer) ) + 
	( (rand3 & (1 << ((i/div_layer3)%cells_per_layer) )) >> ( (i/div_layer3)%cells_per_layer) ) + 
	( (rand4 & (1 << ((i/div_layer4)%cells_per_layer) )) >> ( (i/div_layer4)%cells_per_layer) ) + 
	( (rand5 & (1 << ((i/div_layer5)%cells_per_layer) )) >> ( (i/div_layer5)%cells_per_layer) ); 
    }
    //gettimeofday(&t[5],NULL);
    rdtscl(t[5]);
    float time_us_cpu = tstamp_to_us(t[4], t[5]);  

    tsqbar_cpu += time_us_cpu * time_us_cpu;
    tbar_cpu += time_us_cpu;

    /*
    int mismatches = 0;
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
    */
  }
  
  float trms = sqrt(tsqbar - (tbar*tbar)/(1.0*n))/(n-1.0);
  tbar = tbar/(1.0*n);
  printf("#GPU timing: %5.2f +- %5.2f us\n", tbar, trms);

  float trms_cpu = sqrt(tsqbar_cpu - (tbar_cpu*tbar_cpu)/(1.0*n))/(n-1.0);
  tbar_cpu = tbar_cpu/(1.0*n);
  printf("#CPU timing: %5.2f +- %5.2f us\n", tbar_cpu, trms_cpu);
  //printf("# ndiffs = %d\n", ndiffs);
  
  return 0;
}
