#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "main.h"
#include "svtsim_functions.h"
#include "functionkernel.h"

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

// typedefs for shorthand

typedef thrust::tuple<unsigned int, unsigned int>     DataPair;

                // (layer, out1, out2, out3)
typedef thrust::tuple<unsigned int, unsigned int,
		      unsigned int, unsigned int>     UnpackTuple;

typedef thrust::device_vector<unsigned int>::iterator IntIterator;
typedef thrust::tuple<IntIterator, IntIterator,
		      IntIterator, IntIterator>       IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple>           ZipIterator;

struct Hitmap {
    int hitmap[6];
};

typedef thrust::device_vector<Hitmap> HitmapVector;

#define COLUMNS 3 //var for testkernel

#define ROWS 2    //var for testkernel

#define N_BLOCKS 1
#define N_THREADS_PER_BLOCK 16

#define CUDA_TIMING

#define DEBUG

#define MAX(x,y) ((x)>(y) ? (x):(y))

// CUDA timer macros
cudaEvent_t c_start, c_stop;
inline void CTSTART() {
#ifdef CUDA_TIMING
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, 0);
#endif
}

inline void CTSTOP(const char *file) {
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

// CUDA variables
int *ids, *out1, *out2, *out3;
int *d_ids, *d_out1, *d_out2, *d_out3;
unsigned int *d_data_in;

long sizeW;

#define gf_mask_gpu(x) (d_gf_maskdata[(x)])

__constant__ int d_gf_maskdata[33] = {
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

void GPU_Init(int n_words)
{
    sizeW = sizeof(int) * n_words;
    
    ids  = (int *)malloc(sizeW);
    out1 = (int *)malloc(sizeW);
    out2 = (int *)malloc(sizeW);
    out3 = (int *)malloc(sizeW);

    cudaMalloc((void **)&d_data_in, sizeW);
    cudaMalloc((void **)&d_ids, sizeW);
    cudaMalloc((void **)&d_out1, sizeW);
    cudaMalloc((void **)&d_out2, sizeW);
    cudaMalloc((void **)&d_out3, sizeW);
}

void GPU_Destroy()
{
    free(ids);
    free(out1);
    free(out2);
    free(out3);
    
    cudaFree(d_data_in);
    cudaFree(d_ids);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_out3);
}

template <typename Vector>
void print(const char *title, const Vector &v)
{
    std::cout << title << ": ";
    for(size_t i = 0; i < v.size(); i++)
	std::cout << v[i] << " ";
    std::cout << "\n";
}


// unpacking function itself. unpack the passed-in int to the tuple.
struct unpacker : public thrust::unary_function<DataPair, UnpackTuple> {

    /* parallel word_decode kernel.
       each word is decoded and layer (id) and output values are set.
       we only use 3 output arrays since depending on the layer,
         we only need 3 different values. this saves allocating/copying empty arrays
       format (out1, out2, out3):
         id <  XFT_LYR: zid, lcl, hit
	 id == XFT_LYR: crv, crv_sign, phi
	 id == IP_LYR: sector, amroad, 0
	 id == EE_LYR: ee_word
    */

    __host__ __device__
    UnpackTuple operator()(DataPair t) {
	unsigned int word = thrust::get<0>(t);
	unsigned int prev_word = thrust::get<1>(t);
	
	unsigned int val1 = 0, val2 = 0, val3 = 0;

	int ee, ep, lyr;

	lyr = -999; /* Any invalid numbers != 0-7 */

	/*
	if (word > gf_mask_gpu(SVT_WORD_WIDTH)) {
	    //printf("gf_iword_decode: Input data is larger than the maximum SVT word");
	    //return SVTSIM_GF_ERR;
	    return;
	}
	*/

	/* check if this is a EP or EE word */
	ee = (word >> SVT_EE_BIT)  & gf_mask_gpu(1);
	ep = (word >> SVT_EP_BIT)  & gf_mask_gpu(1);

	// check if this is the second XFT word
	//int prev_word = (idx==0) ? 0 : words[idx-1];
	bool xft = ((prev_word >> SVT_LYR_LSB) & gf_mask_gpu(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;

	if (ee && ep) { /* End of Event word */
	    val1 = word; // ee_word
	    //*parity_in = (word >> SVT_PAR_BIT) & gf_mask_gpu(1);
	    lyr = EE_LYR;
	} else if (ee) { /* only EE bit ON is error condition */
	    //*err |= (1 << UNKNOWN_ERR);
	    lyr = EE_LYR; /* We have to check */
	} else if (ep) { /* End of Packet word */
	    lyr = EP_LYR;
	    val1 = 6; // sector
	    /*   *sector = (word >> SVT_SECT_LSB)  & gf_mask_gpu(SVT_SECT_WIDTH); */
	    val2 = word  & gf_mask_gpu(AMROAD_WORD_WIDTH); // amroad
	} else if (xft) { /* Second XFT word */
	    val1 = (word >> SVT_CRV_LSB)  & gf_mask_gpu(SVT_CRV_WIDTH); // crv
	    val2 = (word >> (SVT_CRV_LSB + SVT_CRV_WIDTH))  & gf_mask_gpu(1); // crv_sign
	    val3 = word & gf_mask_gpu(SVT_PHI_WIDTH); // phi
	    lyr = XFT_LYR_2;
	} else { /* SVX hits or the first XFT word */
	    lyr = (word >> SVT_LYR_LSB)  & gf_mask_gpu(SVT_LYR_WIDTH);
	    if (lyr == XFT_LYRID) lyr = XFT_LYR; // probably don't need - stp
	    val1 = (word >> SVT_Z_LSB)  & gf_mask_gpu(SVT_Z_WIDTH); // zid
	    val2 = (word >> SVT_LCLS_BIT) & gf_mask_gpu(1); // lcl
	    val3 = word & gf_mask_gpu(SVT_HIT_WIDTH); // hit
	}

	return thrust::make_tuple(lyr,val1,val2,val3);
    }
};

struct isNewRoad : public thrust::unary_function<unsigned int, bool> {
    __host__ __device__ bool operator()(const unsigned int &id) {
	//return id == EP_LYR;
	return id == EP_LYR || id == EE_LYR;
    }
};

struct isNewHit : public thrust::unary_function<unsigned int, bool> {
    __host__ __device__ bool operator()(const unsigned int &id) {
	return id < XFT_LYR || id == XFT_LYR_2;
    }
};

struct isNewEvt : public thrust::unary_function<unsigned int, bool> {
    __host__ __device__ bool operator()(const unsigned int &id) {
	return id == EE_LYR;
    }
};

struct lhitToHitmap : public thrust::unary_function<DataPair, Hitmap> {
    __host__ __device__ Hitmap operator()(const DataPair &t) {
	int layer = thrust::get<0>(t);
	int lhit  = thrust::get<1>(t);
	if (layer == XFT_LYR_2) layer = XFT_LYR;
	Hitmap h;
	for (int i=0; i<=XFT_LYR; i++)
	    h.hitmap[i] = layer == i ? lhit : 0;
	return h;
    }
};



struct tupleSecond {// : public thrust::unary_function<T, bool> {
    template <typename T>
    __host__ __device__ bool operator()(const T &t) {
	return thrust::get<1>(t);
    }
};


struct isEqualLayer : public thrust::binary_function<unsigned int, unsigned int, bool>
{
    __host__ __device__ bool operator()(const unsigned int &a, const unsigned int &b) {
	return a == b || ((a == XFT_LYR || a == XFT_LYR_2) && (b == XFT_LYR || b == XFT_LYR_2));
    }
};

struct layerHitMultiply
{
    template <typename T>
    __host__ __device__ T operator()(const T &a, const T &b) {
	//return a * (b>1 ? b:1);
	return MAX(a,1) * MAX(b,1);
    }
};


struct hitmapAccumulate
{
    template <typename T>
    __host__ __device__ T operator()(const T& a, const T& b) {
	Hitmap r;
	for (int i=0; i<=XFT_LYR; i++)
	    r.hitmap[i] = MAX(a.hitmap[i], b.hitmap[i]);
	return r;
    }
};


struct hitmapComb : public thrust::unary_function<DataPair,Hitmap>
{
    Hitmap *d_hitmap;
    hitmapComb(Hitmap *_hm) : d_hitmap(_hm) {} // constructor

    template <typename T>
    __host__ __device__ Hitmap operator()(const T& t) {
	unsigned int road = thrust::get<0>(t);
	unsigned int ic   = thrust::get<1>(t) -1;
	Hitmap hm = d_hitmap[road];
	Hitmap r;

	for (int i=0; i<=XFT_LYR; i++) {
	    int nh = hm.hitmap[i];
	    if (nh == 0) {
		r.hitmap[i] = 0;
	    } else {
		r.hitmap[i] = ic % nh + 1;
		ic /= nh;
	    }
	}

	return r;
    }
};

struct hitmapAbsoluteIndices : public thrust::unary_function<DataPair,Hitmap>
{
    Hitmap *d_hitmap;
    unsigned int *d_road_indices;
    hitmapAbsoluteIndices(Hitmap *_hm, unsigned int *_ri) : d_hitmap(_hm), d_road_indices(_ri) {} // constructor

    template <typename T>
    __host__ __device__ Hitmap operator()(const T& t) {
	unsigned int road = thrust::get<0>(t);
	Hitmap hm_c = thrust::get<1>(t);
	Hitmap hm = d_hitmap[road];
	int offset = d_road_indices[road];
	Hitmap r;

	int ihits = 0;
	for (int i=0; i<=XFT_LYR; i++) {
	    int ih = hm_c.hitmap[i] - 1;
	    if (i == XFT_LYR) ih += 1+ih; // to account for unused first XFT word
	    if (ih < 0) r.hitmap[i] = -1;
	    else r.hitmap[i] = offset + ihits + ih;
	    ihits += hm.hitmap[i];
	}
	return r;
    }
};

// BinaryPredicate for the head flag segment representation
// equivalent to thrust::not2(thrust::project2nd<int,int>()));
template <typename HeadFlagType>
struct head_flag_predicate : public thrust::binary_function<HeadFlagType,HeadFlagType,bool>
{
    __host__ __device__
    bool operator()(HeadFlagType left, HeadFlagType right) const {
	return !right;
    }
};

struct fill_tf_gpu
{
    tf_arrays_t tf; // pointer in device memory
    fill_tf_gpu(tf_arrays_t _tf) : tf(_tf) {} // constructor

    template <typename Tuple>
    __device__ void operator()(Tuple t) {
	unsigned int id      = thrust::get<0>(t);
	unsigned int id_next = thrust::get<1>(t);
	unsigned int out1    = thrust::get<2>(t);
	unsigned int out2    = thrust::get<3>(t);
	unsigned int out3    = thrust::get<4>(t);
	unsigned int evt     = thrust::get<5>(t);
	unsigned int road    = thrust::get<6>(t);
	unsigned int rhit    = thrust::get<7>(t) -1;
	unsigned int lhit    = thrust::get<8>(t) -1;

	// SVX Data
	if (id < XFT_LYR) {
	    int zid = out1;
	    int lcl = out2;
	    int hit = out3;

	    tf->evt_hit[evt][road][id][lhit] = hit;
	    tf->evt_hitZ[evt][road][id][lhit] = zid;
	    tf->evt_lcl[evt][road][id][lhit] = lcl;
	    tf->evt_lclforcut[evt][road][id][lhit] = lcl;
	    tf->evt_layerZ[evt][road][id] = zid;

	    if (rhit == 0) {
		atomicOr(&tf->evt_zid[evt][road], zid & gf_mask_gpu(GF_SUBZ_WIDTH));
	    } else if (id_next == XFT_LYR) {
		atomicOr(&tf->evt_zid[evt][road], (zid & gf_mask_gpu(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH);
	    }
	    
	    //tf->evt_nhits[evt][road][id]++;
	    atomicAdd(&tf->evt_nhits[evt][road][id], 1);
	    
	    // Error Checking
	    if (lhit == MAX_HIT) tf->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);
	    //if (id < id_last) tf->evt_err[evt][road] |= (1 << OUTORDER_BIT);
	} else if (id == XFT_LYR) {
	    // we ignore but leave here to not trigger 'else' case - stp
	} else if (id == XFT_LYR_2) {
	    id = XFT_LYR; // for XFT_LYR_2 kludge - stp
	    int crv      = out1;
	    int crv_sign = out2;
	    int phi      = out3;
	    
	    tf->evt_crv[evt][road][lhit] = crv;
	    tf->evt_crv_sign[evt][road][lhit] = crv_sign;
	    tf->evt_phi[evt][road][lhit] = phi;
	    
	    //tf->evt_nhits[evt][road][id]++;
	    atomicAdd(&tf->evt_nhits[evt][road][id], 1);
	    
	    // Error Checking
	    if (lhit == MAX_HIT) tf->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);
	    //if (id < id_last) tf->evt_err[evt][road] |= (1 << OUTORDER_BIT);
	} else if (id == EP_LYR) {
	    int sector = out1;
	    int amroad = out2;
	    
	    tf->evt_cable_sect[evt][road] = sector;
	    tf->evt_sect[evt][road] = sector;
	    tf->evt_road[evt][road] = amroad;
	    tf->evt_err_sum[evt] |= tf->evt_err[evt][road];

	    //tf->evt_nroads[evt]++;
	    atomicAdd(&tf->evt_nroads[evt], 1);
	    
	    if (road > MAXROAD) {
		;
		//printf("The limit on the number of roads fitted by the TF is %d\n",MAXROAD);
		//printf("You reached that limit evt->road = %d\n",road);
	    }

	    //for (id = 0; id <= XFT_LYR; id++)
	    //  tf->evt_nhits[evt][road][id] = 0;
	    
	    //tf->evt_err[evt][road] = 0;
	    //tf->evt_zid[evt][road] = -1;

	} else if (id == EE_LYR) {
	    int ee_word = out1;
	    
	    tf->evt_ee_word[evt] = ee_word;
	    //tf->totEvts++;
	    atomicAdd(&tf->totEvts, 1);
	} else {
	    //printf("Error INV_DATA_BIT: layer = %u\n", id);
	    tf->evt_err[evt][road] |= (1 << INV_DATA_BIT);
	}

    }
};

__global__ void
k_word_decode(int N, unsigned int *words, int *ids, int *out1, int *out2, int *out3)
{
    /* parallel word_decode kernel.
       each word is decoded and layer (id) and output values are set.
       we only use 3 output arrays since depending on the layer,
         we only need 3 different values. this saves allocating/copying empty arrays
       format (out1, out2, out3):
         id <  XFT_LYR: zid, lcl, hit
	 id == XFT_LYR: crv, crv_sign, phi
	 id == IP_LYR: sector, amroad, 0
	 id == EE_LYR: ee_word
    */
    
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > N) return;

    //ids[idx] = idx; return;

    int word = words[idx];
    int ee, ep, lyr;

    lyr = -999; /* Any invalid numbers != 0-7 */

    out1[idx] = 0;
    out2[idx] = 0;
    out3[idx] = 0;
    
    if (word > gf_mask_gpu(SVT_WORD_WIDTH)) {
        //printf("gf_iword_decode: Input data is larger than the maximum SVT word");
        //return SVTSIM_GF_ERR;

	ids[idx] = lyr;
	return;
    }

    /* check if this is a EP or EE word */
    ee = (word >> SVT_EE_BIT)  & gf_mask_gpu(1);
    ep = (word >> SVT_EP_BIT)  & gf_mask_gpu(1);

    // check if this is the second XFT word
    int prev_word = (idx==0) ? 0 : words[idx-1];
    bool xft = ((prev_word >> SVT_LYR_LSB) & gf_mask_gpu(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;

    if (ee && ep) { /* End of Event word */
        out1[idx] = word; // ee_word
        //*parity_in = (word >> SVT_PAR_BIT) & gf_mask_gpu(1);
        lyr = EE_LYR;
    } else if (ee) { /* only EE bit ON is error condition */
        //*err |= (1 << UNKNOWN_ERR);
        lyr = EE_LYR; /* We have to check */
    } else if (ep) { /* End of Packet word */
        lyr = EP_LYR;
        out1[idx] = 6; // sector
        /*   *sector = (word >> SVT_SECT_LSB)  & gf_mask_gpu(SVT_SECT_WIDTH); */
        out2[idx] = word  & gf_mask_gpu(AMROAD_WORD_WIDTH); // amroad
    } else if (xft) { /* Second XFT word */
        out1[idx] = (word >> SVT_CRV_LSB)  & gf_mask_gpu(SVT_CRV_WIDTH); // crv
        out2[idx] = (word >> (SVT_CRV_LSB + SVT_CRV_WIDTH))  & gf_mask_gpu(1); // crv_sign
        out3[idx] = word & gf_mask_gpu(SVT_PHI_WIDTH); // phi
        lyr = XFT_LYR_2;
    } else { /* SVX hits or the first XFT word */
        lyr = (word >> SVT_LYR_LSB)  & gf_mask_gpu(SVT_LYR_WIDTH);
        if (lyr == XFT_LYRID) lyr = XFT_LYR; // probably don't need - stp
        out1[idx] = (word >> SVT_Z_LSB)  & gf_mask_gpu(SVT_Z_WIDTH); // zid
        out2[idx] = (word >> SVT_LCLS_BIT) & gf_mask_gpu(1); // lcl
        out3[idx] = word & gf_mask_gpu(SVT_HIT_WIDTH); // hit
    }

    ids[idx] = lyr;
}

void scan_threads_per_block_fep(int n_words, unsigned int *words, int *ids, int *out1, int *out2, int *out3)
{

    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);

    int step = 2; //64;
    int n_threads_max;
    float elapsedTime, totalTime = 0;
    int i;
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    n_threads_max = deviceProp.maxThreadsPerBlock;
    printf("n_threads_max = %d\n", n_threads_max);

    // call once without timing to be sure GPU is initialized
    i = n_threads_max;
    k_word_decode <<<(n_words+i-1)/i, i>>>
	    (n_words, d_data_in, d_ids, d_out1, d_out2, d_out3);

    for (i=1; i<n_threads_max; i+=step) {

	int j, n_iter = 10;
	totalTime = 0;
	for (j = 0; j < n_iter; j++) {
		    
	    cudaEventRecord(c_start, 0);
	    k_word_decode <<<(n_words+i-1)/i, i>>>
		(n_words, d_data_in, d_ids, d_out1, d_out2, d_out3);
	    cudaEventRecord(c_stop, 0);
	    cudaEventSynchronize(c_stop);

	    cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
	    elapsedTime *= 1000.0; // ms to us
	    totalTime += elapsedTime;
	}

	elapsedTime = totalTime / n_iter;
	
	FILE *outFile = fopen("threads_scan.txt", "a");
	fprintf(outFile, "%d\t%f\n", i, elapsedTime);
	fclose(outFile);

	if (i==1) i = 0;

    }

}

void launchFepUnpackKernel(tf_arrays_t tf, unsigned int *data_in, int n_words)
{

    /* initializing arrays */
    int ie, id;
    tf->totEvts = 0;

    for (ie = 0; ie < NEVTS; ie++) {
        tf->evt_nroads[ie] = 0;
        tf->evt_err_sum[ie] = 0;

        for (id = 0; id <= NSVX_PLANE; id++) {
            tf->evt_layerZ[ie][tf->evt_nroads[ie]][id] = 0;
        }

        for (id = 0; id <= XFT_LYR; id++) {
            tf->evt_nhits[ie][tf->evt_nroads[ie]][id] = 0;
        }

        tf->evt_err[ie][tf->evt_nroads[ie]] = 0;
        //tf->evt_zid[ie][tf->evt_nroads[ie]] = -1;
	tf->evt_zid[ie][tf->evt_nroads[ie]] = 0; // we need to or these - stp
        //    printf("tf->evt_nroads[%d] = %d, tf->evt_zid[%d][tf->evt_nroads[%d]] = %d\n", ie, tf->evt_nroads[ie], ie, ie, tf->evt_zid[ie][tf->evt_nroads[ie]]);
        //}
    }

    

    CTSTART();
    // Copy data to the Device
    cudaMemcpy(d_data_in, data_in, sizeW, cudaMemcpyHostToDevice);
    CTSTOP("copyWordsToDevice_CUDA");
    
    //scan_threads_per_block_fep(n_words, d_data_in, d_ids, d_out1, d_out2, d_out3);
    
    CTSTART();
    k_word_decode <<<(n_words+N_THREADS_PER_BLOCK-1)/N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>
	(n_words, d_data_in, d_ids, d_out1, d_out2, d_out3);
    CTSTOP("k_word_decode");

    // Copy output to the Host
    CTSTART();
    cudaMemcpy(ids, d_ids, sizeW, cudaMemcpyDeviceToHost);
    cudaMemcpy(out1, d_out1, sizeW, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, d_out2, sizeW, cudaMemcpyDeviceToHost);
    cudaMemcpy(out3, d_out3, sizeW, cudaMemcpyDeviceToHost);
    CTSTOP("k_word_decode_copyToHost");

    
    //////////////////// also do with THRUST
    // input data
    thrust::device_vector<unsigned int> d_vec(n_words); // this would be done in the GPU_Init()
    CTSTART();
    thrust::copy(data_in, data_in + n_words, d_vec.begin());
    CTSTOP("copyWordsToDevice_thrust");

    // output vectors
    thrust::device_vector<unsigned int> d_idt(n_words);
    thrust::device_vector<unsigned int> d_out1t(n_words);
    thrust::device_vector<unsigned int> d_out2t(n_words);
    thrust::device_vector<unsigned int> d_out3t(n_words);

    // unpack
    CTSTART();
    thrust::transform(
	thrust::make_zip_iterator(thrust::make_tuple(d_vec.begin(), d_vec.begin()-1)),
	thrust::make_zip_iterator(thrust::make_tuple(d_vec.end(), d_vec.end()-1)),
	thrust::make_zip_iterator(thrust::make_tuple(d_idt.begin(), d_out1t.begin(),
						     d_out2t.begin(), d_out3t.begin())),
	unpacker());
    CTSTOP("thrust_unpacker");
    
    // copy to CPU for verification
    thrust::host_vector<unsigned int> h_test0 = d_idt;
    thrust::host_vector<unsigned int> h_test1 = d_out1t;
    thrust::host_vector<unsigned int> h_test2 = d_out2t;
    thrust::host_vector<unsigned int> h_test3 = d_out3t;

    /*
    int ndiff = 0;
    for (int i=0; i<n_words; i++) {
	if (h_test0[i] != ids[i]) ndiff++;
	if (h_test1[i] != out1[i]) ndiff++;
	if (h_test2[i] != out2[i]) ndiff++;
	if (h_test3[i] != out3[i]) ndiff++;
    }
    printf("ndiff = %d\n", ndiff);
    printf("nmatch = %d\n", 4*n_words - ndiff);
    */
    
    //// fill nevt, nroad, nhit arrays
    //// want to restart counting according to evt > road > layer > hit
    
    thrust::device_vector<unsigned int> d_evt(n_words);
    thrust::device_vector<unsigned int> d_road(n_words);
    thrust::device_vector<unsigned int> d_rhit(n_words);
    thrust::device_vector<unsigned int> d_lhit(n_words);

    CTSTART();
    thrust::transform(d_idt.begin(), d_idt.end(), d_road.begin(), isNewRoad());
    CTSTOP("scans_singleTransform");

    CTSTART();
    thrust::exclusive_scan(
	thrust::make_transform_iterator(d_idt.begin(), isNewEvt()),
	thrust::make_transform_iterator(d_idt.end(),   isNewEvt()),
	d_evt.begin());
    CTSTOP("scans_exclusive_scan");

    CTSTART();
    thrust::exclusive_scan_by_key(
	d_evt.begin(), d_evt.end(), // keys
	thrust::make_transform_iterator(d_idt.begin(), isNewRoad()), // vals
	d_road.begin());
    CTSTOP("scans_exclusive_scan_by_key");

    CTSTART();
    thrust::inclusive_scan_by_key(
	d_road.begin(), d_road.end(), // keys
	thrust::make_transform_iterator(d_idt.begin(), isNewHit()), //vals
	d_rhit.begin());
    CTSTOP("scans_inclusive_scan_by_key_rhit");

    CTSTART();
    thrust::inclusive_scan_by_key(
	d_idt.begin(), d_idt.end(), // keys
	thrust::make_transform_iterator(d_idt.begin(), isNewHit()), //vals
	d_lhit.begin(),
	isEqualLayer()); // binary predicate
    CTSTOP("scans_inclusive_scan_by_key_lhit");

    //// alternate method of segmenting based on flags instead of scans
    thrust::device_vector<unsigned int> d_evt_flag(n_words);
    thrust::device_vector<unsigned int> d_road_flag(n_words);
    //thrust::device_vector<unsigned int> d_rhit_flag(n_words);
    //thrust::device_vector<unsigned int> d_lhit_flag(n_words);

    thrust::transform(d_idt.begin(), d_idt.end(), d_evt_flag.begin(), isNewEvt());
    thrust::transform(d_idt.begin(), d_idt.end(), d_road_flag.begin(), isNewRoad());
    CTSTART();
    // can do key-based operations on flags instead of scans
    thrust::inclusive_scan_by_key(
	d_road_flag.begin(), d_road_flag.end(), // keys
	thrust::make_transform_iterator(d_idt.begin(), isNewHit()), //vals
	d_rhit.begin(),
	head_flag_predicate<unsigned int>());
    CTSTOP("scan_inclusive_scan_by_key_rhit_flags");

    //// calculate number of combinations per road
    // for the size of these, only need n_roads, but might be slower(?) to wait for
    // that result to come back to CPU
    thrust::device_vector<unsigned int> d_roadKey(n_words); // will soon only take up n_roads
    thrust::device_vector<unsigned int> d_ncomb(n_words);   // will soon only take up n_roads
    
    CTSTART();
    size_t n_roads = thrust::reduce_by_key(
	d_road.begin(), d_road.end(), // keys
	d_lhit.begin(),               // vals
	d_roadKey.begin(),            // keys output
	d_ncomb.begin(),              // vals output
	thrust::equal_to<int>(),      // binary predicate
	layerHitMultiply()            // binary operator
    ).first - d_roadKey.begin();  // new output size
    CTSTOP("reduce_by_key");


#ifdef DEBUG
    for (int i=0; i<n_words; i++) {
	unsigned int evt = d_evt[i];
	unsigned int road = d_road[i];
	unsigned int rhit = d_rhit[i];
	unsigned int lhit = d_lhit[i];
	printf("%.6x\tevt = %d\troad = %d\trhit = %d\tlayer = %d\tlhit = %d\tout=(%.6x,%.6x,%.6x)\n", data_in[i], evt, road, rhit, h_test0[i], lhit, h_test1[i], h_test2[i], h_test3[i]);
    }
#endif

#ifdef DEBUG
    for (int i=0; i<n_roads; i++) {
	unsigned int road = d_roadKey[i];
	unsigned int ncomb = d_ncomb[i];
	printf("road %d has %d combinations\n", road, ncomb);
    }
#endif
    
    // get global road offset indices
    CTSTART();
    thrust::device_vector<unsigned int> d_road_indices(n_roads);
    thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(1), d_road_flag.begin())),
		    thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(n_words+1), d_road_flag.end())),
		    thrust::make_zip_iterator(thrust::make_tuple(d_road_indices.begin(), thrust::constant_iterator<int>(0))),
		    tupleSecond());
    CTSTOP("road_indices");
#ifdef DEBUG
    print("road_indices", d_road_indices);
#endif
    
    CTSTART();
    thrust::device_vector<unsigned int> d_ncomb_scan(n_roads);
    thrust::inclusive_scan(d_ncomb.begin(), d_ncomb.begin() + n_roads, d_ncomb_scan.begin());
    //unsigned int n_combs = thrust::reduce(d_ncomb.begin(), d_ncomb.end());
    unsigned int n_combs = d_ncomb_scan.back();
#ifdef DEBUG
    printf("total combinations: %d\n", n_combs);
#endif
    
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
    
    /* // can also do with a scan but will take longer
    thrust::inclusive_scan_by_key(
	d_indices.begin(), d_indices.end(),
	thrust::constant_iterator<int>(1),
	d_indices.begin());
    */
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
    CTSTOP("indices");

#ifdef DEBUG
    print("ncomb_scan", d_ncomb_scan);
#endif

#ifdef DEBUG
    printf("indices: ");
    for (int i=0; i<n_combs; i++) {
	unsigned int index = d_indices[i];
	printf("%d ", index);
    }
    printf("\n");
#endif
    
    CTSTART();
    HitmapVector d_hitmap(n_roads);

    // faster way would be to copy_if (layer,lhit) according to the isNewLayer flag to grab the last lhit
    // then reduce_by_key(road) and write into tuple (no collision -> no MAX() needed)
    thrust::reduce_by_key(
	d_road.begin(), d_road.end(), // keys
	thrust::make_transform_iterator(
	    thrust::make_zip_iterator(thrust::make_tuple(d_idt.begin(), d_lhit.begin())),
	    lhitToHitmap()),           // vals
	d_roadKey.begin(),            // keys output
	d_hitmap.begin(),             // vals output
	thrust::equal_to<int>(),      // binary predicate
	hitmapAccumulate());          // binary operator

    CTSTOP("hitmaps");

#ifdef DEBUG
    for (int i=0; i<n_roads; i++) {
	Hitmap t = d_hitmap[i];
	printf("road = %d, hitmap = (%d, %d, %d, %d, %d, %d)\n", i,
	       t.hitmap[0], t.hitmap[1], t.hitmap[2],
	       t.hitmap[3], t.hitmap[4], t.hitmap[5]);
    }
#endif
    
    // get combination hitmaps
    CTSTART();
    HitmapVector d_hitmap_combs(n_combs);
    thrust::transform(
	thrust::make_zip_iterator(thrust::make_tuple(d_indices_road.begin(), d_indices.begin())),
	thrust::make_zip_iterator(thrust::make_tuple(d_indices_road.end(), d_indices.end())),
	d_hitmap_combs.begin(),
	hitmapComb(thrust::raw_pointer_cast(&d_hitmap[0])));
    CTSTOP("hitmapComb");

#ifdef DEBUG
    for (int i=0; i<n_combs; i++) {
	unsigned int road = d_indices_road[i];
	uint comb = d_indices[i];
	Hitmap t = d_hitmap_combs[i];
	printf("road = %d, comb = %d, hitmap = (%d, %d, %d, %d, %d, %d)\n", road, comb,
	       t.hitmap[0], t.hitmap[1], t.hitmap[2],
	       t.hitmap[3], t.hitmap[4], t.hitmap[5]);
    }
#endif
    
    // get absolute hit indices in the word data list
    CTSTART();
    thrust::transform(
	thrust::make_zip_iterator(thrust::make_tuple(d_indices_road.begin(), d_hitmap_combs.begin())),
	thrust::make_zip_iterator(thrust::make_tuple(d_indices_road.end(), d_hitmap_combs.end())),
	d_hitmap_combs.begin(),
	hitmapAbsoluteIndices(
	    thrust::raw_pointer_cast(&d_hitmap[0]),
	    thrust::raw_pointer_cast(&d_road_indices[0])));
    CTSTOP("hitmapCombAbs");

#ifdef DEBUG
    printf("\nabsolute combinations:\n");
    for (int i=0; i<n_combs; i++) {
	unsigned int road = d_indices_road[i];
	uint comb = d_indices[i];
	Hitmap t = d_hitmap_combs[i];
	printf("road = %d, comb = %d, hitmap = (%d, %d, %d, %d, %d, %d)\n", road, comb,
	       t.hitmap[0], t.hitmap[1], t.hitmap[2],
	       t.hitmap[3], t.hitmap[4], t.hitmap[5]);
    }
#endif
    
    ///////////////// fill tf on GPU

    // Copy tf to the Device
    long tfSize = sizeof(struct tf_arrays);
    tf_arrays_t d_tf;
    cudaMalloc((void **)&d_tf, tfSize);
    cudaMemcpy(d_tf, tf, tfSize, cudaMemcpyHostToDevice);

    CTSTART();
    
    thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
        d_idt.begin(), d_idt.begin()+1, d_out1t.begin(), d_out2t.begin(), d_out3t.begin(),
        d_evt.begin(), d_road.begin(), d_rhit.begin(), d_lhit.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
        d_idt.end(), d_idt.end()+1, d_out1t.end(), d_out2t.end(), d_out3t.end(),
        d_evt.end(), d_road.end(), d_rhit.end(), d_lhit.end())),
      fill_tf_gpu(d_tf));

    CTSTOP("thrust_fill");


    /*
    // Copy tf to the Host
    CTSTART();
    cudaMemcpy(tf, d_tf, tfSize, cudaMemcpyDeviceToHost);
    CTSTOP("copyTFtoHost");

    // for informational purposes
    CTSTART();
    cudaMemcpy(d_tf, tf, tfSize, cudaMemcpyHostToDevice);
    CTSTOP("copyTFtoDevice");
    */
    
    cudaFree(d_tf);

    
    ///////////////// now fill tf (gf_fep_unpack)

    for (ie = 0; ie < NEVTS; ie++) {
        tf->evt_zid[ie][tf->evt_nroads[ie]] = -1; // because we set it to 0 for GPU version
    }

    CTSTART();
    
    int id_last = -1;
    int evt = EVT;

    unsigned int *data = (unsigned int *) data_in;

    for (int i = 0; i < n_words; i++) {
        id = ids[i];

	bool gf_xft = 0;
	if (id == XFT_LYR_2) { // compatibility - stp
	    id = XFT_LYR; 
	    gf_xft = 1;
	}
	
	int nroads = tf->evt_nroads[evt];
	int nhits = tf->evt_nhits[evt][nroads][id];
	
	// SVX Data
	if (id < XFT_LYR) {
	    int zid = out1[i];
	    int lcl = out2[i];
	    int hit = out3[i];

	    tf->evt_hit[evt][nroads][id][nhits] = hit;
	    tf->evt_hitZ[evt][nroads][id][nhits] = zid;
	    tf->evt_lcl[evt][nroads][id][nhits] = lcl;
	    tf->evt_lclforcut[evt][nroads][id][nhits] = lcl;
	    tf->evt_layerZ[evt][nroads][id] = zid;

	    if (tf->evt_zid[evt][nroads] == -1) {
		tf->evt_zid[evt][nroads] = zid & gf_mask(GF_SUBZ_WIDTH);
	    } else {
		tf->evt_zid[evt][nroads] = (((zid & gf_mask(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH)
					    + (tf->evt_zid[evt][nroads] & gf_mask(GF_SUBZ_WIDTH)));
	    }
	    
	    nhits = ++tf->evt_nhits[evt][nroads][id];
	
	    // Error Checking
	    if (nhits == MAX_HIT) tf->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
	    if (id < id_last) tf->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);
	} else if (id == XFT_LYR && gf_xft == 0) {
	    // we ignore - stp
	} else if (id == XFT_LYR && gf_xft == 1) {
	    
	    int crv = out1[i];
	    int crv_sign = out2[i];
	    int phi = out3[i];

	    tf->evt_crv[evt][nroads][nhits] = crv;
	    tf->evt_crv_sign[evt][nroads][nhits] = crv_sign;
	    tf->evt_phi[evt][nroads][nhits] = phi;

	    nhits = ++tf->evt_nhits[evt][nroads][id];
	
	    // Error Checking
	    if (nhits == MAX_HIT) tf->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
	    if (id < id_last) tf->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);
	} else if (id == EP_LYR) {
	    int sector = out1[i];
	    int amroad = out2[i];
	    
	    tf->evt_cable_sect[evt][nroads] = sector;
	    tf->evt_sect[evt][nroads] = sector;
	    tf->evt_road[evt][nroads] = amroad;
	    tf->evt_err_sum[evt] |= tf->evt_err[evt][nroads];

	    nroads = ++tf->evt_nroads[evt];
	    
	    if (nroads > MAXROAD) {
		printf("The limit on the number of roads fitted by the TF is %d\n",MAXROAD);
		printf("You reached that limit evt->nroads = %d\n",nroads);
	    }

	    for (id = 0; id <= XFT_LYR; id++)
		tf->evt_nhits[evt][nroads][id] = 0;
	
	    tf->evt_err[evt][nroads] = 0;
	    tf->evt_zid[evt][nroads] = -1;

	    id = -1; id_last = -1;
	} else if (id == EE_LYR) {
	    int ee_word = out1[i];
	    
	    tf->evt_ee_word[evt] = ee_word;
	    tf->totEvts++;
	    evt++;

	    id = -1; id_last = -1;
	} else {
	    printf("Error INV_DATA_BIT: layer = %u\n", id);
	    tf->evt_err[evt][nroads] |= (1 << INV_DATA_BIT);
	}
	id_last = id;

    } //end loop on input words

    CTSTOP("fill_CPU");

}



///////////////////////////////////////////////////////////////////////////////////////////

__global__ void
kTestKernel_tf(tf_arrays_t tf)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    tf->gf_emsk = 1; // test - stp

}

void launchTestKernel_tf(tf_arrays_t tf, unsigned int *data_in, int n_words)
{

    printf("sizeof(tf_arrays) = %u\n", sizeof(struct tf_arrays));
    printf("sizeof(tf) = %u\n", sizeof(tf));

    //printf("%d\n", tf);
    //printf("before kernel: %u\n", tf->dummy);
    printf("before kernel: %u\n", tf->gf_emsk);

    long tfSize = sizeof(struct tf_arrays);

    // Allocate device tf array
    tf_arrays_t d_tf;
    cudaMalloc((void **)&d_tf, tfSize);

    // Copy tf to the Device
    cudaMemcpy(d_tf, tf, tfSize, cudaMemcpyHostToDevice);

    // Kernel
    kTestKernel_tf<<<n_words, 1>>>(d_tf);

    // Copy tf to the Host
    cudaMemcpy(tf, d_tf, tfSize, cudaMemcpyDeviceToHost);

    printf("after kernel: %u\n", tf->gf_emsk);
}

///////////////////////////////////////////////////////////////////////////////////////////


__global__ void
kFepComb(unsigned int *data_out, unsigned int *data_in)
{

    /*
        This function calculates all the combinations of hits given a certain road.
        For each road we can have multiple hits per layer.
        the input of this function is the set of "evt_" arrays, the output is:

      int fep_ncmb[NEVTS][MAXROAD];
      int fep_hit[NEVTS][MAXROAD][MAXCOMB][NSVX_PLANE];
      int fep_phi[NEVTS][MAXROAD][MAXCOMB];
      int fep_crv[NEVTS][MAXROAD][MAXCOMB];
      int fep_lcl[NEVTS][MAXROAD][MAXCOMB];
      int fep_lclforcut[NEVTS][MAXROAD][MAXCOMB];
      int fep_hitmap[NEVTS][MAXROAD][MAXCOMB];
      int fep_zid[NEVTS][MAXROAD];
      int fep_road[NEVTS][MAXROAD];
      int fep_sect[NEVTS][MAXROAD];
      int fep_cable_sect[NEVTS][MAXROAD];
      int fep_err[NEVTS][MAXROAD][MAXCOMB][MAXCOMB5H];
      int fep_crv_sign[NEVTS][MAXROAD][MAXCOMB];
      int fep_ncomb5h[NEVTS][MAXROAD][MAXCOMB];
      int fep_hitZ[NEVTS][MAXROAD][MAXCOMB][NSVX_PLANE];
      int fep_nroads[NEVTS];
      int fep_ee_word[NEVTS];
      int fep_err_sum[NEVTS];


    */

}

void launchFepComb(unsigned int *data_res, unsigned int *data_in)
{
    kFepComb <<< N_BLOCKS, N_THREADS_PER_BLOCK>>>(data_res, data_in);
}

/////////////////////////////////////////////////////////////////////////////////


__global__ void
kFit(int *fit_fit_dev, int *fep_ncmb_dev)
{
    /*
        This function, for each road, for each combination:
        - retrieves the correct constant set, based on
         tf->fep_hitmap[ie][ir][ic], tf->fep_lcl[ie][ir][ic], tf->fep_zid[ie][ir]
        - performs the scalar product.

        It handles the 5/5 tracks as well (for each 5/5 track, 5 4/5 fits are run, and only the
        best is kept).

        The inputs of this function are the fep arrays, the output are:
      long long int fit_fit[NEVTS][6][MAXROAD][MAXCOMB][MAXCOMB5H];
      int fit_err[NEVTS][MAXROAD][MAXCOMB][MAXCOMB5H];
      int fit_err_sum[NEVTS];

        All the arrays needed by the function (constants, etc..) need to be stored on
        memory easily accessible by the GPU.
    */

    /*
       int ir, ic, ip, ih, ihit, il, i;
       int rc = 1;
       int hit[SVTNHITS];
       long long int coeff[NFITTER][SVTNHITS];
       int coe_addr, int_addr; // Address for coefficients and intercept
       int mka_addr; // Address for MKADDR memory
       long long int theintcp = 0;
       int sign_crv = 0;
       int which, lwhich;
       int iz;
       int newhitmap;
       int g = 0;
       int p0[6], ix[7];
       int ie;

       //  struct fep_out *fep;
       //struct fit_out *trk;
       int map[7][7] = {
         { 0, 1, 2, 3, -1, 4, 5 }, // 01235
         { 0, 1, 2, -1, 3, 4, 5 }, // 01245
         { 0, 1, -1, 2, 3, 4, 5 }, // 01345
         { 0, -1, 1, 2, 3, 4, 5 }, // 02345
         { -1, 0, 1, 2, 3, 4, 5 }, // 12345
         { 0, 1, 2, 3, -1, 4, 5 }, // (??)
         { 0, 1, 2, 3, -1, 4, 5 }  // (??)
       };
    */

    /* --------- Executable starts here ------------ */


//the following are just test...
//    ie =blockIdx.x;
//    if(ie != 0) ie = ie + (MAXROAD-1);
//    for(ir = 0; ir < MAXROAD; ir++) {
//      i = ie+ir;
//      fit_fit_dev[i] = fep_ncmb_dev[i];
//    }
//
//      fit_fit_dev[ie] = fep_nroads_dev[ie];
//
//    int x = blockIdx.x;
//
//    i=0;
//    for(ir = 0; ir < 100; ir++) {
//        i = blockIdx.x + 100*ir;
//
//      fit_fit_dev[i] = fep_ncmb_dev[i];
//    }

}

void launchFitKernel(int *fit_fit_dev, int *fep_ncmb_dev)
{

    kFit <<< NEVTS, 1>>>(fit_fit_dev, fep_ncmb_dev);
}


//////////////////////////////////////////////////////////


__global__ void kTestKernel(int *a, int *b, int *c)

{

    int x = blockIdx.x;
    int y;
    int i;
    for (y = 0; y < ROWS; y++) {
        i = x + (COLUMNS * y);
        c[i] = a[i] + b[i];
    }


}


void launchTestKernel(int *dev_a, int *dev_b, int *dev_c)
{

    kTestKernel <<< COLUMNS, 1>>>(dev_a, dev_b, dev_c);
}

