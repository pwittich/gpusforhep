// nvcc -arch=sm_20 -I ../include/ -o expand expand.cu ../src/format.cpp ../src/random.cpp ../src/mgpucontext.cpp


#include "moderngpu.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

//#define DEBUG
#define CUDA_TIMING

using namespace mgpu;

// CUDA timer macros
cudaEvent_t c_start, c_stop;
inline void CTSTART()
{
#ifdef CUDA_TIMING
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, 0);
#endif
}

inline void CTSTOP(const char *file)
{
#ifdef CUDA_TIMING
    cudaEventRecord(c_stop, 0);
    cudaEventSynchronize(c_stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
    elapsedTime *= 1000.0; // ms to us
    char filePath[100];
    sprintf(filePath, "timer/%s.txt", file);
    FILE *outFile = fopen(filePath, "a");
    if (outFile != NULL) {
        fprintf(outFile, "%f\n", elapsedTime);
        fclose(outFile);
    } else {
        printf("Warning: cannot open %s\n", filePath);
    }
#endif
}

inline float CTSTOPGET()
{
#ifdef CUDA_TIMING
    cudaEventRecord(c_stop, 0);
    cudaEventSynchronize(c_stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
    elapsedTime *= 1000.0; // ms to us
    return elapsedTime;
#else
    return 0;
#endif
}

// Copy IntervalExpand and IntervalGather host function from kernels/intervalmove.cuh and modify to
// expose the tuning structure.

////////////////////////////////////////////////////////////////////////////////
// IntervalExpand

template<typename Tuning, typename IndicesIt, typename ValuesIt, typename OutputIt>
MGPU_HOST void TuningIntervalExpand(int moveCount, IndicesIt indices_global, 
			      ValuesIt values_global, int intervalCount, OutputIt output_global,
			      CudaContext& context) {

    //const int NT = 128;
    //const int VT = 7;
    //typedef LaunchBoxVT<NT, VT> Tuning;
    int2 launch = Tuning::GetLaunchParams(context);

    int NV = launch.x * launch.y;
    int numBlocks = MGPU_DIV_UP(moveCount + intervalCount, NV);

    // Partition the input and output sequences so that the load-balancing
    // search results in a CTA fit in shared memory.
    MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsUpper>(
	mgpu::counting_iterator<int>(0), moveCount, indices_global, 
	intervalCount, NV, 0, mgpu::less<int>(), context);

    KernelIntervalExpand<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(
	moveCount, indices_global, values_global, intervalCount,
	partitionsDevice->get(), output_global);
}

////////////////////////////////////////////////////////////////////////////////
// IntervalGather

template<typename Tuning, typename GatherIt, typename IndicesIt, typename InputIt,
	typename OutputIt>
MGPU_HOST void TuningIntervalGather(int moveCount, GatherIt gather_global, 
	IndicesIt indices_global, int intervalCount, InputIt input_global,
	OutputIt output_global, CudaContext& context) {

    //const int NT = 128;
    //const int VT = 7;
    //typedef LaunchBoxVT<NT, VT> Tuning;
    int2 launch = Tuning::GetLaunchParams(context);

    int NV = launch.x * launch.y;
    int numBlocks = MGPU_DIV_UP(moveCount + intervalCount, NV);
	
    MGPU_MEM(int) partitionsDevice = MergePathPartitions<MgpuBoundsUpper>(
	mgpu::counting_iterator<int>(0), moveCount, indices_global,
	intervalCount, NV, 0, mgpu::less<int>(), context);

    KernelIntervalMove<Tuning, true, false>
	<<<numBlocks, launch.x, 0, context.Stream()>>>(moveCount, gather_global,
	(const int*)0, indices_global, intervalCount, input_global,
        partitionsDevice->get(), output_global);
}



template <typename Vector>
void print(const char *title, const Vector &v)
{
    std::cout << title << ": ";
    for (size_t i = 0; i < v.size(); i++)
        std::cout << v[i] << " ";
    std::cout << "\n";
}

void MGPU(int NROADS, int NCOMBS, CudaContext &context)
{
    int Counts[NROADS];
    std::fill(Counts, Counts + NROADS, NCOMBS);
    //const int Counts[10] = {3, 8, 10, 4, 20, 8, 29, 2, 10, 20};

    /*
    const int Gather[NROADS] = {
        1, 1, 1, 1
    };
    */

#ifdef DEBUG
    printf("\n-- MGPU --\n");
    printf("Interval counts:\n");
    PrintArray(Counts, NROADS, "%4d", 10);

    //printf("\nInterval gather:\n");
    //PrintArray(Gather, NROADS, "%4d", 10);
#endif

    MGPU_MEM(int) countsDevice = context.Malloc(Counts, NROADS);
    //MGPU_MEM(int) gatherDevice = context.Malloc(Gather, NROADS);
    int total = mgpu::Scan(countsDevice->get(), NROADS, context);

    MGPU_MEM(int) dataDevice1 = context.Malloc<int>(total);
    MGPU_MEM(int) dataDevice2 = context.Malloc<int>(total);

    IntervalExpand(total, countsDevice->get(), mgpu::counting_iterator<int>(0), NROADS,
                   dataDevice1->get(), context);

    IntervalGather(total, mgpu::step_iterator<int>(1, 0), //gatherDevice->get(),
                   countsDevice->get(), NROADS, mgpu::counting_iterator<int>(0),
                   dataDevice2->get(), context);

#ifdef DEBUG
    printf("\nroad:\n");
    PrintArray(*dataDevice1, "%4d", 10);
    printf("\ncomb:\n");
    PrintArray(*dataDevice2, "%4d", 10);
#endif

}

template<typename Tuning>
void MGPU_Tuning(int NROADS, int NCOMBS, CudaContext &context)
{
    int Counts[NROADS];
    std::fill(Counts, Counts + NROADS, NCOMBS);
    //const int Counts[10] = {3, 8, 10, 4, 20, 8, 29, 2, 10, 20};

    /*
    const int Gather[NROADS] = {
        1, 1, 1, 1
    };
    */

#ifdef DEBUG
    printf("\n-- MGPU --\n");
    printf("Interval counts:\n");
    PrintArray(Counts, NROADS, "%4d", 10);

    //printf("\nInterval gather:\n");
    //PrintArray(Gather, NROADS, "%4d", 10);
#endif

    MGPU_MEM(int) countsDevice = context.Malloc(Counts, NROADS);
    //MGPU_MEM(int) gatherDevice = context.Malloc(Gather, NROADS);
    int total = mgpu::Scan(countsDevice->get(), NROADS, context);

    MGPU_MEM(int) dataDevice1 = context.Malloc<int>(total);
    MGPU_MEM(int) dataDevice2 = context.Malloc<int>(total);

    TuningIntervalExpand<Tuning>(total, countsDevice->get(), mgpu::counting_iterator<int>(0), NROADS,
                   dataDevice1->get(), context);

    TuningIntervalGather<Tuning>(total, mgpu::step_iterator<int>(1, 0), //gatherDevice->get(),
                   countsDevice->get(), NROADS, mgpu::counting_iterator<int>(0),
                   dataDevice2->get(), context);

#ifdef DEBUG
    printf("\nroad:\n");
    PrintArray(*dataDevice1, "%4d", 10);
    printf("\ncomb:\n");
    PrintArray(*dataDevice2, "%4d", 10);
#endif

}

void MGPU_scan(int NROADS, CudaContext &context)
{
    int Counts[NROADS];
    std::fill(Counts, Counts + NROADS, 1);

#ifdef DEBUG
    printf("\n-- MGPU_scan --\n");
    printf("Input:\n");
    PrintArray(Counts, NROADS, "%4d", 10);
#endif

    MGPU_MEM(int) countsDevice = context.Malloc(Counts, NROADS);
    int total = mgpu::Scan(countsDevice->get(), NROADS, context);

#ifdef DEBUG
    printf("total: %d\n", total);
    printf("output:\n");
    PrintArray(*countsDevice, "%4d", 10);
#endif

}

void MGPU_reduce(int NROADS, CudaContext &context)
{
    int Counts[NROADS];
    std::fill(Counts, Counts + NROADS, 1);

#ifdef DEBUG
    printf("\n-- MGPU_scan --\n");
    printf("Input:\n");
    PrintArray(Counts, NROADS, "%4d", 10);
#endif

    MGPU_MEM(int) countsDevice = context.Malloc(Counts, NROADS);
    int total = mgpu::Reduce(countsDevice->get(), NROADS, context);

#ifdef DEBUG
    printf("total: %d\n", total);
    printf("output:\n");
    PrintArray(*countsDevice, "%4d", 10);
#endif

}

    
void Thrust(int NROADS, int NCOMBS)
{
    int Counts[NROADS];
    std::fill(Counts, Counts + NROADS, NCOMBS);
    //const int Counts[10] = {3, 8, 10, 4, 20, 8, 29, 2, 10, 20};

    thrust::device_vector<unsigned int> d_ncomb(Counts, Counts + NROADS);

#ifdef DEBUG
    printf("\n-- Thrust --\n");
    printf("input: ");
    for (int i = 0; i < NROADS; i++) {
        unsigned int ncomb = d_ncomb[i];
        printf(" %d", ncomb);
    }
    printf("\n");
#endif

    thrust::device_vector<unsigned int> d_ncomb_scan(NROADS);
    thrust::inclusive_scan(d_ncomb.begin(), d_ncomb.begin() + NROADS, d_ncomb_scan.begin());
    unsigned int n_combs = d_ncomb_scan.back();

    // get the combination indices. might be able to do this better with an iterator, like
    // https://github.com/thrust/thrust/blob/master/examples/repeated_range.cu
    thrust::device_vector<unsigned int> d_indices(n_combs);
    thrust::lower_bound(d_ncomb_scan.begin(), d_ncomb_scan.end(),
                        thrust::counting_iterator<unsigned int>(1),
                        thrust::counting_iterator<unsigned int>(n_combs + 1),
                        d_indices.begin());

    thrust::device_vector<unsigned int> d_indices_road(d_indices);
#ifdef DEBUG
    print("indices_road", d_indices_road);
#endif


    thrust::inclusive_scan_by_key(
	d_indices.begin(), d_indices.end(),
	thrust::constant_iterator<int>(1),
	d_indices.begin());

/*    
    // different way to do it; doesn't quite work when input is not constant
    thrust::gather(d_indices.begin(), d_indices.end(),
                   d_ncomb_scan.begin(),
                   d_indices.begin());

    thrust::transform(d_indices.begin(), d_indices.end(),
                      thrust::constant_iterator<int>(*d_ncomb_scan.begin()),
                      d_indices.begin(),
                      thrust::minus<int>());

    
    thrust::transform(thrust::counting_iterator<int>(1),
                      thrust::counting_iterator<int>(n_combs + 1),
                      d_indices.begin(),
                      d_indices.begin(),
                      thrust::minus<int>());
*/
    
#ifdef DEBUG
    printf("indices: ");
    for (int i = 0; i < n_combs; i++) {
        unsigned int index = d_indices[i];
        printf("%d ", index);
    }
    printf("\n");
#endif

}

void Thrust_scan(int NROADS)
{

    int Counts[NROADS];
    std::fill(Counts, Counts + NROADS, 1);

    thrust::device_vector<unsigned int> d_ncomb(Counts, Counts + NROADS);

#ifdef DEBUG
    printf("\n-- Thrust --\n");
    printf("input:");
    for (int i = 0; i < NROADS; i++) {
        unsigned int ncomb = d_ncomb[i];
        printf(" %d", ncomb);
    }
    printf("\n");
#endif

    unsigned int front = d_ncomb.back();
    thrust::exclusive_scan(d_ncomb.begin(), d_ncomb.end(), d_ncomb.begin());
    unsigned int n_combs = front + d_ncomb.back();

#ifdef DEBUG
    printf("total: %d\n", n_combs);
    print("n_comb_scan", d_ncomb);
#endif
}

void Thrust_reduce(int NROADS)
{

    int Counts[NROADS];
    std::fill(Counts, Counts + NROADS, 1);

    thrust::device_vector<unsigned int> d_ncomb(Counts, Counts + NROADS);

#ifdef DEBUG
    printf("\n-- Thrust --\n");
    printf("input:");
    for (int i = 0; i < NROADS; i++) {
        unsigned int ncomb = d_ncomb[i];
        printf(" %d", ncomb);
    }
    printf("\n");
#endif

    unsigned int n_combs = thrust::reduce(d_ncomb.begin(), d_ncomb.end());

#ifdef DEBUG
    printf("total: %d\n", n_combs);
    print("n_comb", d_ncomb);
#endif
}

template <typename Tuning>
void doTiming(CudaContext &context, int NCOMBS=16) {

    int N_AVG = 100; // number of timing measurements to average per data point
    int nroad_lo = 0;
    int nroad_hi = 100000;
    int nroad_step = 1000;
    int nroad_iter = (nroad_hi - nroad_lo) / nroad_step;

    int2 launch = Tuning::GetLaunchParams(context);
    int NT = launch.x;
    int VT = launch.y;
	
    char filePath[100];
//    if (NT != 128 || VT != 7)
	sprintf(filePath, "timer_bench2/NCOMBS_%i_NT_%d_VT_%d.txt", NCOMBS, NT, VT);
//    else
//	sprintf(filePath, "timer_tmp/NCOMBS_%i.txt", NCOMBS);

    printf("output: %s\n", filePath);

    FILE *outFile = fopen(filePath, "a");
    if (outFile == NULL)
	printf("Warning: cannot open %s\n", filePath);
    
    for (int ir = 0; ir <= nroad_iter; ir++) {
	int NROADS = nroad_lo + nroad_step * ir;
	if (NROADS == 0) NROADS = 1;
	float t_mgpu = 0;
	float t_thrust = 0;
	for (int i = 0; i < N_AVG; i++) {
	    CTSTART();
	    MGPU_Tuning<Tuning>(NROADS, NCOMBS, context);
	    //MGPU(NROADS, NCOMBS, context);
	    //MGPU_scan(NROADS, context);
	    //MGPU_reduce(NROADS, context);
	    t_mgpu += CTSTOPGET();
	    CTSTART();
	    Thrust(NROADS, NCOMBS);
	    //Thrust_scan(NROADS);
	    //Thrust_reduce(NROADS);
	    t_thrust += CTSTOPGET();
	}
	t_mgpu /= N_AVG;
	t_thrust /= N_AVG;

	fprintf(outFile, "%d\t%f\t%f\n", NROADS, t_mgpu, t_thrust);

    }

    fclose(outFile);

}

int main(int argc, char **argv)
{
    // Initialize a CUDA device on the default stream.
    //ContextPtr context = mgpu::CreateCudaDevice(argc, argv, true);
    ContextPtr context = mgpu::CreateCudaDevice(0, argv, true);

    int NROADS = 4; //10;
    int NCOMBS = 32;

    MGPU(NROADS, NCOMBS, *context);
    Thrust(NROADS, NCOMBS);

    MGPU_scan(NROADS, *context);
    Thrust_scan(NROADS);

    MGPU_reduce(NROADS, *context);
    Thrust_reduce(NROADS);
    

#ifdef CUDA_TIMING
    
    int c;
    while ((c = getopt(argc, argv, "c:h")) != -1) {
	switch (c) {
	case 'c':
	    NCOMBS = atoi(optarg);
	    break;
	case 'h':
	    fprintf(stderr, "usage: %s [cnv]\n", argv[0]);
	    fprintf(stderr, "  c = # combinations \n");
	    return 0;
	}
    }
    printf("NCOMBS = %d\n", NCOMBS);

    typedef LaunchBoxVT<128,  5> Tuning1;
    typedef LaunchBoxVT<128,  7> Tuning2; // default
    typedef LaunchBoxVT<128, 11> Tuning3;
    typedef LaunchBoxVT<128, 15> Tuning4;
    typedef LaunchBoxVT<128, 19> Tuning5;
    typedef LaunchBoxVT<128, 23> Tuning6;
    typedef LaunchBoxVT<128, 27> Tuning7;
    typedef LaunchBoxVT<256,  5> Tuning8;
    typedef LaunchBoxVT<256,  7> Tuning9;
    typedef LaunchBoxVT<256, 11> Tuning10;
    typedef LaunchBoxVT<256, 15> Tuning11;
    typedef LaunchBoxVT<256, 19> Tuning12;
    typedef LaunchBoxVT<256, 23> Tuning13;
    typedef LaunchBoxVT<256, 27> Tuning14;

    doTiming<Tuning1>(*context, NCOMBS);
    doTiming<Tuning2>(*context, NCOMBS);
    doTiming<Tuning3>(*context, NCOMBS);
    doTiming<Tuning4>(*context, NCOMBS);
    doTiming<Tuning5>(*context, NCOMBS);
    doTiming<Tuning6>(*context, NCOMBS);
    doTiming<Tuning7>(*context, NCOMBS);
    doTiming<Tuning8>(*context, NCOMBS);
    doTiming<Tuning9>(*context, NCOMBS);
    doTiming<Tuning10>(*context, NCOMBS);
    doTiming<Tuning11>(*context, NCOMBS);
    doTiming<Tuning12>(*context, NCOMBS);

    //doTiming(*context, NCOMBS);

#endif
    
    return 0;
}
