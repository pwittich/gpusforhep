#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <sys/types.h>

#include <sys/time.h>

#include <cuda.h>
#include "svtsim_functions.h"
#include "device_functions.h"


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>

// typedefs for shorthand

typedef thrust::tuple<unsigned int, unsigned int>     DataTuple;

typedef thrust::tuple<unsigned int, unsigned int,
          unsigned int, unsigned int>     UnpackTuple;

typedef thrust::device_vector<unsigned int>::iterator IntIterator;
typedef thrust::tuple<IntIterator, IntIterator,
          IntIterator, IntIterator>       IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple>           ZipIterator;

#define MAX(x,y) ((x)>(y) ? (x):(y))

#define TIMING

#define MY_CUDA_CHECK( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define MY_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

extern "C" {
  int gf_init(tf_arrays_t* ptr_tf);
  int svtsim_fconread(tf_arrays_t tf);
  int gf_fep_unpack(tf_arrays_t tf, int n_words_in, void* data); 
  int gf_fep_comb(tf_arrays_t tf);
  int gf_comparator(tf_arrays_t tf);
  void svtsim_cable_addwords(svtsim_cable_t *cable, unsigned int *word, int nword);
  void svtsim_cable_addword(svtsim_cable_t *cable, unsigned int word);
  void svtsim_cable_copywords(svtsim_cable_t *cable, unsigned int *word, int nword);
  svtsim_cable_t * svtsim_cable_new(void);
  int gf_fit(tf_arrays_t tf);
}

// CUDA timer macros
cudaEvent_t c_start, c_stop;
inline void start_time() {
#ifdef TIMING
  cudaEventCreate(&c_start);
  cudaEventCreate(&c_stop);
  cudaEventRecord(c_start, 0);
#endif
}

inline void stop_time(const char *msg) {
#ifdef TIMING  
  cudaEventRecord(c_stop, 0);
  cudaEventSynchronize(c_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
  printf("Time to %s: %.3f ms\n", msg, elapsedTime);
#endif
}

#define gf_mask_GPU(x) (gf_maskdata_GPU[(x)])
__device__ static int gf_maskdata_GPU[] = {
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

#define gf_mask3_GPU(x) (gf_maskdata3_GPU[(x)])
__device__ static long long int gf_maskdata3_GPU[] = {
  0x000000000000ULL,
  0x000000000001ULL, 0x000000000003ULL, 0x000000000007ULL, 0x00000000000fULL,
  0x00000000001fULL, 0x00000000003fULL, 0x00000000007fULL, 0x0000000000ffULL,
  0x0000000001ffULL, 0x0000000003ffULL, 0x0000000007ffULL, 0x000000000fffULL,
  0x000000001fffULL, 0x000000003fffULL, 0x000000007fffULL, 0x00000000ffffULL,
  0x00000001ffffULL, 0x00000003ffffULL, 0x00000007ffffULL, 0x0000000fffffULL,
  0x0000001fffffULL, 0x0000003fffffULL, 0x0000007fffffULL, 0x000000ffffffULL,
  0x000001ffffffULL, 0x000003ffffffULL, 0x000007ffffffULL, 0x00000fffffffULL,
  0x00001fffffffULL, 0x00003fffffffULL, 0x00007fffffffULL, 0x0000ffffffffULL,
  0x0001ffffffffULL, 0x0003ffffffffULL, 0x0007ffffffffULL, 0x000fffffffffULL,
  0x001fffffffffULL, 0x003fffffffffULL, 0x007fffffffffULL, 0x00ffffffffffULL,
  0x01ffffffffffULL, 0x03ffffffffffULL, 0x07ffffffffffULL, 0x0fffffffffffULL,
  0x1fffffffffffULL, 0x3fffffffffffULL, 0x7fffffffffffULL, 0xffffffffffffULL
};



struct unpacker : public thrust::unary_function<DataTuple, UnpackTuple> {

  __device__ UnpackTuple operator()(DataTuple t) {
    unsigned int word = thrust::get<0>(t);
    unsigned int prev_word = thrust::get<1>(t);
    unsigned int val1 = 0, val2 = 0, val3 = 0;

    int ee, ep, lyr;

    lyr = -999; /* Any invalid numbers != 0-7 */

    /* check if this is a EP or EE word */
    ee = (word >> SVT_EE_BIT)  & gf_mask_GPU(1);
    ep = (word >> SVT_EP_BIT)  & gf_mask_GPU(1);

    // check if this is the second XFT word
    bool xft = ((prev_word >> SVT_LYR_LSB) & gf_mask_GPU(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;

    if (ee && ep) { /* End of Event word */
      val1 = word; // ee_word
      lyr = EE_LYR;
    } else if (ee) { /* only EE bit ON is error condition */
      lyr = EE_LYR; /* We have to check */
    } else if (ep) { /* End of Packet word */
      lyr = EP_LYR;
      val1 = 6; // sector
      val2 = word  & gf_mask_GPU(AMROAD_WORD_WIDTH); // amroad
    } else if (xft) { /* Second XFT word */
      val1 = (word >> SVT_CRV_LSB)  & gf_mask_GPU(SVT_CRV_WIDTH); // crv
      val2 = (word >> (SVT_CRV_LSB + SVT_CRV_WIDTH))  & gf_mask_GPU(1); // crv_sign
      val3 = word & gf_mask_GPU(SVT_PHI_WIDTH); // phi
      lyr = XFT_LYR_2;
    } else { /* SVX hits or the first XFT word */
      lyr = (word >> SVT_LYR_LSB)  & gf_mask_GPU(SVT_LYR_WIDTH);
      if (lyr == XFT_LYRID) lyr = XFT_LYR; // probably don't need - stp
      val1 = (word >> SVT_Z_LSB)  & gf_mask_GPU(SVT_Z_WIDTH); // zid
      val2 = (word >> SVT_LCLS_BIT) & gf_mask_GPU(1); // lcl
      val3 = word & gf_mask_GPU(SVT_HIT_WIDTH); // hit
    }

    return thrust::make_tuple(lyr,val1,val2,val3);
  }
};


struct isNewRoad : public thrust::unary_function<unsigned int, bool> {
  __host__ __device__ bool operator()(const unsigned int &id) {
    return id == EP_LYR;
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

struct isEqualLayer : public thrust::binary_function<unsigned int, unsigned int, bool> {
  __host__ __device__ bool operator()(const unsigned int &a, const unsigned int &b) {
    return a == b || ((a == XFT_LYR || a == XFT_LYR_2) && (b == XFT_LYR || b == XFT_LYR_2));
  }
};

struct layerHitMultiply {
  template <typename T>
  __host__ __device__ T operator()(const T &a, const T &b) {
    return MAX(a,1) * MAX(b,1);
  }
};


struct fill_tf_gpu {
  int *totEvts; // pointer in device memory
  struct evt_arrays *tf; // pointer in device memory
  __device__ fill_tf_gpu(struct evt_arrays *_tf, int *_totEvts) : tf(_tf), totEvts(_totEvts) {
    *totEvts = 0;
  } // constructor

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
        atomicOr(&tf->evt_zid[evt][road], zid & gf_mask_GPU(GF_SUBZ_WIDTH));
      } else if (id_next == XFT_LYR) {
        atomicOr(&tf->evt_zid[evt][road], (zid & gf_mask_GPU(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH);
      }

      atomicAdd(&tf->evt_nhits[evt][road][id], 1);

      // Error Checking
      if (lhit == MAX_HIT) 
        tf->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);

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

      atomicAdd(&tf->evt_nhits[evt][road][id], 1);

      // Error Checking
      if (lhit == MAX_HIT) 
        tf->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);
    } else if (id == EP_LYR) {
      int sector = out1;
      int amroad = out2;

      tf->evt_cable_sect[evt][road] = sector;
      tf->evt_sect[evt][road] = sector;
      tf->evt_road[evt][road] = amroad;
      tf->evt_err_sum[evt] |= tf->evt_err[evt][road];

      atomicAdd(&tf->evt_nroads[evt], 1);

    } else if (id == EE_LYR) {
      int ee_word = out1;

      tf->evt_ee_word[evt] = ee_word;
      atomicAdd(totEvts, 1);
    } else {
      tf->evt_err[evt][road] |= (1 << INV_DATA_BIT);
    }
  }
};



__device__ int svtsim_whichFit_full_GPU(int layerMask, int lcMask) {

   switch (layerMask & 0x1f) {
   case 0x0f: /* 0123 */
     return 0;
   case 0x17: /* 0124 */
     return 1;
   case 0x1b: /* 0134 */
     return 2;
   case 0x1d: /* 0234 */
     return 3;
   case 0x1e: /* 1234 */
     return 4;
   case 0x1f: /* 01234 - this is the fun one to be careful with */
     if(lcMask == 0)
       return 2; /* use 0134 if we have no LC */
     else if (lcMask == 0x1)
       return 4;
     else if (lcMask == 0x2)
       return 3;
     else if (lcMask == 0x3)
       return 3;
     else if (lcMask == 0x4)
       return 2;
     else if (lcMask == 0x5)
       return 2;
     else if (lcMask == 0x6)
       return 2;
     else if (lcMask == 0x7)
       return 2;
     else if (lcMask == 0x8)
       return 1;
     else if (lcMask == 0x9)
       return 1;
     else if (lcMask == 0xa)
       return 1;
     else if (lcMask == 0xb)
       return 1;
     else if (lcMask == 0xc)
       return 2;
     else if (lcMask == 0xd)
       return 2;
     else if (lcMask == 0xe)
       return 2;
     else if (lcMask == 0xf)
       return 2;
     else  /* If we have LC on outer layer just use 0123 */
       return 0;
   default:
     return 0;

  }
}

__device__ int svtsim_whichFit_GPU(struct extra_data* edata_dev, int zin, int layerMask, int lcMask) {

   int which0 = 0, which = 0;
   if (zin<0 || zin>=SVTSIM_NBAR) zin = 0;
   which0 = svtsim_whichFit_full_GPU(layerMask, lcMask);
   which = edata_dev->whichFit[zin][which0];

   return which;
}


__device__ int  svtsim_get_gfMkAddr_GPU(struct extra_data* edata_dev, int *d, int nd, int d0) {

   /* 
      d0 = iaddr
      
   */
   int j;
   int md = 0x4000;
   int iz, lcl, hit;

   if (d0+nd>md) nd = md-d0;
   for (j = 0; j<nd; j++) {
     int i = j+d0;
     int word = 0xffff, intcp = 0, coeff = 0;
     int which;

     iz = i&7, lcl = i>>3 & 0x1f, hit = i>>8 & 0x3f;

     which = svtsim_whichFit_GPU(edata_dev, iz, hit, lcl);
     coeff = iz + which*6;  /* poor choice for illegal iz=6,7, but compatible */
     intcp = which;

     word = coeff<<3 | intcp;
     d[j] = word;
   }
   return nd;
}

__device__  int gf_mkaddr_GPU(struct extra_data* edata_dev, int hitmap, int lclmap, int zmap,
                          int *coe_addr, int *int_addr, int *addr, int *err) {

  int iaddr;
  unsigned int datum = 0;

  if ((hitmap<0) || (hitmap > gf_mask_GPU( NSVX_PLANE + 1 )) || /* + XFT_LYR */
       (lclmap<0) || (lclmap > gf_mask_GPU( NSVX_PLANE )) ||
       (zmap<0)   || (zmap   > gf_mask_GPU( GF_ZID_WIDTH )))
    *err |= ( 1 << SVTSIM_GF_MKADDR_INVALID );

  iaddr = ((zmap & gf_mask_GPU(GF_SUBZ_WIDTH)) + (lclmap<<MADDR_NCLS_LSB) + (hitmap<<MADDR_HITM_LSB));
#define MAXMKA 8192
  if ((iaddr < 0) || (iaddr >= MAXMKA)) return SVTSIM_GF_ERR;

  int ldat = 0;
  svtsim_get_gfMkAddr_GPU(edata_dev, &ldat, 1, iaddr);
  datum = ldat;
    
  *int_addr = datum & gf_mask_GPU(OFF_SUBA_WIDTH);
  *coe_addr = (datum >> OFF_SUBA_WIDTH) & gf_mask_GPU(PAR_ADDR_WIDTH);
  *addr = iaddr;

  return SVTSIM_GF_OK;

}

__device__  int gf_fit_proc_GPU(int hit[], int sign_crv, long long int coeff[], 
                            long long int intcp, long long int *result, int *err) {

  long long int temp = 0;
  int i = 0;

  *result = 0;
  *err = 0;
  for (i = 0; i < SVTNHITS; i++) {
    if (i < NSVX_PLANE) {
      temp += hit[i] * coeff[i];
    } else if (i == HIT_PHI) { /* XFT phi */
      hit[i] = (hit[i]&0x400) ? -((~hit[i]&0x3ff)+1) : (hit[i]&0x3ff);
      temp += hit[i] * coeff[i];
    } else if (i == HIT_CRV) { /* XFT curvature (curv already with sign in fep ) */
      if (sign_crv == 1) { /* if negative bit is set */
        temp -= hit[i] * coeff[i];
      } else {
        temp += hit[i] * coeff[i];
      }
    }
  }
  *result = *result + temp + intcp;
  *result = *result<0 ? -((-*result)>>17) : *result>>17;
  if (*result > 0)
    *result &= gf_mask3_GPU(FIT_DWIDTH);
  else
    *result = -(abs(*result)&gf_mask3_GPU(FIT_DWIDTH));
  return SVTSIM_GF_OK;
}


__device__ int gf_chi2_GPU(long long int chi[], int* trk_err, long long int *chi2) {

  long long int temp = 0;
  long long int chi2memdata = 0;

  *chi2 = 0;

  for (int i=0; i<NCHI; i++) {
    temp = abs(chi[i]);
    if (chi[i] < 0) temp++;

    chi2memdata = temp*temp;
    *chi2 += chi2memdata;

  }

  *chi2 = (*chi2 >> 2);

  if ((*chi2 >> 2) > gf_mask_GPU(CHI_DWIDTH)) {
    *chi2 = 0x7ff;
    *trk_err |= (1 << OFLOW_CHI_BIT);
  }

  return SVTSIM_GF_OK;

}

__device__ int gf_getq_GPU(int lyr_config) {

  int q = 0;

  switch (lyr_config) {
  case 0x01e : /* lcmap = 00000, hitmap = 11110 */
    q = 3;
    break;
  case 0x01d : /* lcmap = 00000, hitmap = 11101 */
    q = 2;
    break;
  case 0x01b : /* lcmap = 00000, hitmap = 11011 */
    q = 1;
    break;
  case 0x017 : /* lcmap = 00000, hitmap = 10111 */
    q = 2;
    break;
  case 0x00f : /* lcmap = 00000, hitmap = 01111 */
    q = 2;
    break;

  case 0x03e : /* lcmap = 00001, hitmap = 11110 */
    q = 2;
    break;
  case 0x03d : /* lcmap = 00001, hitmap = 11101 */
    q = 1;
    break;
  case 0x03b : /* lcmap = 00001, hitmap = 11011 */
    q = 1;
    break;
  case 0x037 : /* lcmap = 00001, hitmap = 10111 */
    q = 1;
    break;
  case 0x02f : /* lcmap = 00001, hitmap = 01111 */
    q = 1;
    break;

  case 0x05e : /* lcmap = 00010, hitmap = 11110 */
    q = 7;
    break;
  case 0x05d : /* lcmap = 00010, hitmap = 11101 */
    q = 1;
    break;
  case 0x05b : /* lcmap = 00010, hitmap = 11011 */
    q = 2;
    break;
  case 0x057 : /* lcmap = 00010, hitmap = 10111 */
    q = 2;
    break;
  case 0x04f : /* lcmap = 00010, hitmap = 01111 */
    q = 2;
    break;
  case 0x09e : /* lcmap = 00100, hitmap = 11110 */
    q = 7;
    break;
  case 0x09d : /* lcmap = 00100, hitmap = 11101 */
    q = 2;
    break;
  case 0x09b : /* lcmap = 00100, hitmap = 11011 */
    q = 1;
    break;
  case 0x097 : /* lcmap = 00100, hitmap = 10111 */
    q = 2;
    break;
  case 0x08f : /* lcmap = 00100, hitmap = 01111 */
    q = 3;
    break;

  case 0x11e : /* lcmap = 01000, hitmap = 11110 */
    q = 7;
    break;
  case 0x11d : /* lcmap = 01000, hitmap = 11101 */
    q = 2;
    break;
  case 0x11b : /* lcmap = 01000, hitmap = 11011 */
    q = 2;
    break;
  case 0x117 : /* lcmap = 01000, hitmap = 10111 */
    q = 1;
    break;
  case 0x10f : /* lcmap = 01000, hitmap = 01111 */
    q = 3;
    break;

  case 0x21e : /* lcmap = 10000, hitmap = 11110 */
    q = 7;
    break;
  case 0x21d : /* lcmap = 10000, hitmap = 11101 */
    q = 2;
    break;
  case 0x21b : /* lcmap = 10000, hitmap = 11011 */
    q = 2;
    break;
  case 0x217 : /* lcmap = 10000, hitmap = 10111 */
    q = 2;
    break;
  case 0x20f : /* lcmap = 10000, hitmap = 01111 */
    q = 1;
    break;

  case 0x0de : /* lcmap = 00110, hitmap = 11110 */
    q = 7;
    break;
  case 0x0dd : /* lcmap = 00110, hitmap = 11101 */
    q = 1;
    break;
  case 0x0db : /* lcmap = 00110, hitmap = 11011 */
    q = 2;
    break;
  case 0x0d7 : /* lcmap = 00110, hitmap = 10111 */
    q = 3;
    break;
  case 0x0cf : /* lcmap = 00110, hitmap = 01111 */
    q = 4;
    break;

  case 0x19e : /* lcmap = 01100, hitmap = 11110 */
    q = 7;
    break;
  case 0x19d : /* lcmap = 01100, hitmap = 11101 */
    q = 2;
    break;
  case 0x19b : /* lcmap = 01100, hitmap = 11011 */
    q = 1;
    break;
  case 0x197 : /* lcmap = 01100, hitmap = 10111 */
    q = 1;
    break;
  case 0x18f : /* lcmap = 01100, hitmap = 01111 */
    q = 3;
    break;


  case 0x31e : /* lcmap = 11000, hitmap = 11110 */
    q = 7;
    break;
  case 0x31d : /* lcmap = 11000, hitmap = 11101 */
    q = 3;
    break;
  case 0x31b : /* lcmap = 11000, hitmap = 11011 */
    q = 3;
    break;
  case 0x317 : /* lcmap = 11000, hitmap = 10111 */
    q = 1;
    break;
  case 0x30f : /* lcmap = 11000, hitmap = 01111 */
    q = 2;
    break;

  case 0x15e : /* lcmap = 01010, hitmap = 11110 */
    q = 7;
    break;
  case 0x15d : /* lcmap = 01010, hitmap = 11101 */
    q = 1;
    break;
  case 0x15b : /* lcmap = 01010, hitmap = 11011 */
    q = 3;
    q = 3;
    break;
  case 0x157 : /* lcmap = 01010, hitmap = 10111 */
    q = 2;
    break;
  case 0x14f : /* lcmap = 01010, hitmap = 01111 */
    q = 4;
    break;

  case 0x25e : /* lcmap = 10010, hitmap = 11110 */
    q = 7;
    break;
  case 0x25d : /* lcmap = 10010, hitmap = 11101 */
    q = 1;
    break;
  case 0x25b : /* lcmap = 10010, hitmap = 11011 */
    q = 2;
    break;
  case 0x257 : /* lcmap = 10010, hitmap = 10111 */
    q = 2;
    break;
  case 0x24f : /* lcmap = 10010, hitmap = 01111 */
    q = 1;
    break;

  case 0x29e : /* lcmap = 10100, hitmap = 11110 */
    q = 7;
    break;
  case 0x29d : /* lcmap = 10100, hitmap = 11101 */
    q = 2;
    break;
  case 0x29b : /* lcmap = 10100, hitmap = 11011 */
    q = 1;
    break;
  case 0x297 : /* lcmap = 10100, hitmap = 10111 */
    q = 2;
    break;
  case 0x28f : /* lcmap = 10100, hitmap = 01111 */
    q = 1;
    break;
  default:
    q = 7;
    break;
  }
  return q;
}

__device__ int gf_gfunc_GPU(int ncomb5h, int icomb5h, int hitmap, int lcmap, int chi2) {

  int lyr_config;
  int gvalue;
  int newhitmap;
  int newlcmap;
  int q = 0;

  if (ncomb5h == 1) {
    newhitmap = hitmap;
    newlcmap = lcmap;
  } else if (ncomb5h == 5) {
    switch (icomb5h) {
    case 0 :     /*  11110 */
      newhitmap = 0x1e;
      newlcmap  = (lcmap & 0x1e);
      break;
    case 1 :     /*  11101 */
      newhitmap = 0x1d;
      newlcmap  = lcmap & 0x1d;
      break;
    case 2 :     /*  11011 */
      newhitmap = 0x1b;
      newlcmap  = lcmap & 0x1b;
      break;
    case 3 :     /*  10111 */
      newhitmap = 0x17;
      newlcmap  = lcmap & 0x17;
      break;
    case 4 :     /*  01111 */
      newhitmap = 0x0f;
      newlcmap  = lcmap & 0x0f;
      break;
    }
  }
  lyr_config = newhitmap + (newlcmap << 5);
  q = gf_getq_GPU(lyr_config);
  gvalue = (q << 4) + ((chi2 & 0x3ff) >> 6);
  return gvalue;
}

__device__ int gf_stword_GPU(int id, int err) {
     /*
       Compose the GF status word in the 7th word from the GF 
       INPUT : err; error summary
       OUTPUT : return the gf_stword

       NOTE: Currently this code does not support the parity error and
             FIFO error.
     */

  int word;

  word = id;

  if ((err>>OFLOW_HIT_BIT)&gf_mask_GPU(1))
    word |= (1<<GFS_OFL_HIT);

  if ((err>>OFLOW_CHI_BIT)&gf_mask_GPU(1))
    word |= (1<<GFS_OFL_CHI);

  if (((err>>UFLOW_HIT_BIT)&gf_mask_GPU(1)) ||
       ((err>>OUTORDER_BIT)&gf_mask_GPU(1)))
    word |= (1<<GFS_INV_DATA);

  return word;

}

__device__ int cal_parity_GPU(int word) {

  int par = 0;

  for (int i=0; i<SVT_WORD_WIDTH; i++)
    par ^= ((word>>i) & gf_mask_GPU(1));

  return par;
}

__device__ int gf_formatter_err_GPU(int err, int cdfmsk, int svtmsk, int eoemsk,
                                    int *eoe, int *cdf, int *svt) {

    /*
       Simulate the board error conditions (CDF-ERR, SVT-ERR and EOE-ERR)
       INPUT: err; error summary.
       cdfmsk; Mask for the CDF-ERR.
       svtmsk; Mask for the SVT-ERR.
       eoemsk; Mask for the EOE-ERR.
       OUTPUT: *eoe; EOE error
       *cdf; CDF error
       *svt; SVT error
       */

  /* --------- Executable starts here ------------ */

  *cdf = 0; /* never turned ON except for the FIFO overflow */
  *svt = 0;
  *eoe = 0;

  for (int i=0; i<= FIT_RESULT_OFLOW_BIT; i++) {
    if ((err>>i)&gf_mask_GPU(1)) {
      if (((svtmsk>>i)&gf_mask_GPU(1)) == 0)
        *svt = 1;
  
      if (i == 0) {
        if (((eoemsk >> PARITY_ERR_BIT) & gf_mask_GPU(1)) == 0) {
          *eoe |= (1<<PARITY_ERR_BIT);
        }
      } else if ((i==2) || (i==3)) {
        if (((eoemsk>>INV_DATA_BIT)&gf_mask_GPU(1)) == 0) {
          *eoe |= (1<<INV_DATA_BIT);
        }
      } else {
        if (((eoemsk>>INT_OFLOW_BIT)&gf_mask_GPU(1)) == 0) {
          *eoe |= (1<<INT_OFLOW_BIT);
        }
      }
    } /* if ((err>>i)&gf_mask_GPU(1))  */

  } /* for (i=0; i<= FIT_RESULT_OFLOW_BIT; i++)  */

  return SVTSIM_GF_OK;


}


__device__ int gf_formatter_GPU(int ie, int ir, int ic, int ich, int chi2, 
                            struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, struct evt_arrays* evt_dev,
                            struct fout_arrays* fout_dev) {

  int it, err;
  int hit_form[NSVX_PLANE];

  int z = 0; /* z should be 6 bits large */
  int gf_stat = 0;

  // atomicAdd returns the old value
  it = atomicAdd(&fout_dev->fout_ntrks[ie], 1);
  
  err = (fep_dev->fep_err[ie][ir] | fit_dev->fit_err[ie][ir][ic][ich]);

  for (int i=0; i<NSVX_PLANE; i++) {
    /* Hit coordinate */
    if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
      if (i != ich) {
        hit_form[i] = fep_dev->fep_hit[ie][ir][ic][i]&gf_mask_GPU(GF_HIT_WIDTH);
        /* Long Cluster bit */
        hit_form[i] += (((fep_dev->fep_hit[ie][ir][ic][i] & 0x4000) ? 1 : 0) << GF_HIT_WIDTH);
        /* Hit existence bit */
        hit_form[i] += (((fep_dev->fep_hitmap[ie][ir][ic]>>i)&gf_mask_GPU(1))<<(GF_HIT_WIDTH+1));
        hit_form[i] = (hit_form[i]&gf_mask_GPU(GF_HIT_WIDTH+2));
      } else 
        hit_form[i] = 0;
    } else {
      hit_form[i] = fep_dev->fep_hit[ie][ir][ic][i]&gf_mask_GPU(GF_HIT_WIDTH);
      /* Long Cluster bit */
      hit_form[i] += (((fep_dev->fep_hit[ie][ir][ic][i] & 0x4000) ? 1 : 0) << GF_HIT_WIDTH);
      /* Hit existence bit */
      hit_form[i] += (((fep_dev->fep_hitmap[ie][ir][ic]>>i)&gf_mask_GPU(1))<<(GF_HIT_WIDTH+1));
      hit_form[i] = (hit_form[i]&gf_mask_GPU(GF_HIT_WIDTH+2));
    }
  }

  if (1) {
    int presentmask;
    int newhitmap;

    if (fep_dev->fep_ncomb5h[ie][ir][ic] == 1) {
     presentmask = fep_dev->fep_hitmap[ie][ir][ic];
    } else if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
      switch (ich) {
      case 0 :     /*  11110 */
        newhitmap = 0x1e;
        break;
      case 1 :     /*  11101 */
        newhitmap = 0x1d;
        break;
      case 2 :     /*  11011 */
        newhitmap = 0x1b;
        break;
      case 3 :     /*  10111 */
        newhitmap = 0x17;
        break;
      case 4 :     /*  01111 */
        newhitmap = 0x0f;
        break;
      }
      presentmask = newhitmap;
    }
    {
      int longmask = presentmask & fep_dev->fep_lcl[ie][ir][ic];
      int goodmask = presentmask & ~longmask;
      int badmask = 0x1f & ~goodmask;
      int badmap[] = {
        0x0,    /* 00000: all layers good */
        0x5,    /* 10000: layer 0 bad */
        0x4,    /* 01000: layer 1 bad */
        0xe,    /* 11000: layers 0,1 bad  (changed from f to e) */
        0x3,    /* 00100: layer 2 bad */
        0xe,    /* 10100: layers 0,2 bad */
        0xb,    /* 01100: layers 1,2 bad */
        0xf,    /* 11100: >2 layers bad */
        0x2,    /* 00010: layer 3 bad */
        0xd,    /* 10010: layers 0,3 bad */
        0xa,    /* 01010: layers 1,3 bad */
        0xf,    /* 11010: >2 layers bad */
        0x8,    /* 00110: layers 2,3 bad */
        0xf,    /* 10110: >2 layers bad */
        0xf,    /* 01110: >2 layers bad */
        0xf,    /* 11110: >2 layers bad */
        0x1,    /* 00001: layer 4 bad */
        0xc,    /* 10001: layers 0,4 bad */
        0x8,    /* 01001: layers 1,4 bad  (oops: doc says 0x9 not 0x8) */
        0xf,    /* 11001: >2 layers bad */
        0x7,    /* 00101: layers 2,4 bad */
        0xf,    /* 10101: >2 layers bad */
        0xf,    /* 01101: >2 layers bad */
        0xf,    /* 11101: >2 layers bad */
        0x6,    /* 00011: layers 3,4 bad */
        0xf,    /* 10011: >2 layers bad */
        0xf,    /* 01011: >2 layers bad */
        0xf,    /* 11011: >2 layers bad */
        0xf,    /* 00111: >2 layers bad */
        0xf,    /* 10111: >2 layers bad */
        0xf,    /* 01111: >2 layers bad */
        0xf     /* 11111: all layers bad! */
      };
    gf_stat = badmap[badmask];
    }
  }
  gf_stat = gf_stword_GPU(gf_stat, err);

  /* output word (25 bits) (from CDFnote 5026)
    4-3-2-1-0-9-8-7-6-5-4-3-2-1-0-9-8-7-6-5-4-3-2-1-0                
  */
  /* 1st word 
    24-23-22-21- 20- 19- 18-17-16-15-14-13- 12-11-10-9-8-7-6-5-4-3-2-1-0 
    --------     1   -  z                  phi     
  */

  /* phi is already formatted by the fitter (13 bits) */
  if (fep_dev->fep_ncomb5h[ie][ir][ic] == 1) {
    z = fep_dev->fep_zid[ie][ir];
  } else if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
    if (ich == 0){
      z = ((fep_dev->fep_hitZ[ie][ir][ic][4]&gf_mask_GPU(GF_SUBZ_WIDTH))<<GF_SUBZ_WIDTH) + (fep_dev->fep_hitZ[ie][ir][ic][1]&gf_mask_GPU(GF_SUBZ_WIDTH));
    } else if (ich == 4){
      z = ((fep_dev->fep_hitZ[ie][ir][ic][3]&gf_mask_GPU(GF_SUBZ_WIDTH))<<GF_SUBZ_WIDTH) + (fep_dev->fep_hitZ[ie][ir][ic][0]&gf_mask_GPU(GF_SUBZ_WIDTH));
    } else {
      z = ((fep_dev->fep_hitZ[ie][ir][ic][4]&gf_mask_GPU(GF_SUBZ_WIDTH))<<GF_SUBZ_WIDTH) + (fep_dev->fep_hitZ[ie][ir][ic][0]&gf_mask_GPU(GF_SUBZ_WIDTH));
    }
  }
  fout_dev->fout_gfword[ie][it][0] = (fit_dev->fit_fit[ie][0][ir][ic][ich] & gf_mask_GPU(OPHI_WIDTH))
                                      + ((z & gf_mask_GPU(GF_ZID_WIDTH)) << OPHI_WIDTH)
                                      + (0 << OBP_ERR_BIT) // we follow the word structure in  http://www-cdf.fnal.gov/internal/upgrades/daq_trig/trigger/svt/BoardDocs/data_words/tracks_bits.html 
                                      + (1<<(OBP_ID_BIT));

  /* 2nd word 
     4-3-2-1-0-9-8   -7-6-5-4-3-2-1-0 -9   -8-7-6-5-4-3-2-1-0 
     24-23-22-21- 20- 19-  18-  17-16-15-14-13- 12-11-  10-9-8-7-6-5-4-3-2-1-0 
     ------------  rID      sign c                       d
     17mo bit di roadID -> 19
     18mo               -> 20
  */
  fout_dev->fout_gfword[ie][it][1] = fit_dev->fit_fit[ie][1][ir][ic][ich]
                                    + (fit_dev->fit_fit[ie][2][ir][ic][ich] << OCVR_LSB)
                                    + ((evt_dev->evt_road[ie][ir] & 0x60000) << 2);

  /* 3rd word 
     4-3-2-1-0-9-8-7 -6-5-4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 
     --------sector   AM road id (17 LSB)
  */
  fout_dev->fout_gfword[ie][it][2] = (evt_dev->evt_road[ie][ir] & gf_mask_GPU(OAMROAD_WIDTH))
                                      + (( fep_dev->fep_cable_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH)) << OSEC_LSB);

  /* 4th word 
     4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 -9-8-7-6-5-4-3-2-1-0 
     ----------x1                   x0
     bit 21 = bit 19 del roadID
     hit = 8 bassi e 2 alti     
  */
  fout_dev->fout_gfword[ie][it][3] = hit_form[0] + (hit_form[1]<<OX1_LSB)
                                      + ((evt_dev->evt_road[ie][ir] & 0x80000) << 1);

  /* 5th word 
     4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 -9-8-7-6-5-4-3-2-1-0 
     ----------x3                   x2
     bit 21 = road ID 20
  */

  fout_dev->fout_gfword[ie][it][4] = hit_form[2] + (hit_form[3]<<OX3_LSB)
                                      + ((evt_dev->evt_road[ie][ir] & 0x100000));

  /* 6th word 
     4-3-2-1-0-9-8-7-6-5-4-3-2-1-0 -9-8-7-6-5-4-3-2-1-0 
     ----------chisq                x4
  */

  fout_dev->fout_gfword[ie][it][5] = hit_form[4] + ((chi2 & gf_mask_GPU(CHI2SUM_WIDTH)) << OCHI2_LSB);

  /* 7th word 
     4-3-2-1 -0-9-8-7-6-5-4-3-2-1-0-9 -8-7-6-5-4-3-2-1-0 
     ------0  TrackFitter status       Track Number                
     Track Num = identificativo della traccia XFT
     phi - 3 bit meno significativi del phi della traccia XFT
  */
  fout_dev->fout_gfword[ie][it][6] = ((fep_dev->fep_phi[ie][ir][ic] >> SVT_TRKID_LSB)
                                      &gf_mask_GPU(SVT_TRKID_WIDTH))
                                      + ((gf_stat & gf_mask_GPU(GF_STAT_WIDTH))<<OSTAT_LSB)
                                      + (1<<SVT_EP_BIT);

  for (int i=0; i<NTFWORDS; i++)
    atomicXor(&fout_dev->fout_parity[ie], cal_parity_GPU(fout_dev->fout_gfword[ie][it][i]));

  return SVTSIM_GF_OK;

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

__global__ void gf_fep_comb_GPU (evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt) {

  int ie, ir;

  int nlyr; /* The number of layers with a hit */
  int ncomb; /* The number of combinations */

  ie = blockIdx.x; // events index
  ir = threadIdx.x; // roads index

  fep_dev->fep_ncmb[ie][ir] = 0;
  fep_dev->fep_zid[ie][ir] = 0;
  fep_dev->fep_road[ie][ir] = 0;
  fep_dev->fep_sect[ie][ir] = 0;
  fep_dev->fep_cable_sect[ie][ir] = 0;
  fep_dev->fep_err[ie][ir] = 0;

  if ( ( ie < maxEvt ) && 
      ( ir < evt_dev->evt_nroads[ie] ) ) {

    ncomb = 1;
    nlyr = 0;
    /* At first, we calculate how many combinations are there */
    for (int id=0; id<(XFT_LYR+1); id++) {
      if (evt_dev->evt_nhits[ie][ir][id] != 0) {
        ncomb *= evt_dev->evt_nhits[ie][ir][id];
        nlyr++;
      }
    }

    if ( nlyr < MINHITS )
      evt_dev->evt_err[ie][ir] |= (1<<UFLOW_HIT_BIT);    

    fep_dev->fep_ncmb[ie][ir] = ncomb;
    atomicOr(&evt_dev->evt_err_sum[ie], evt_dev->evt_err[ie][ir]);

    fep_dev->fep_zid[ie][ir] = (evt_dev->evt_zid[ie][ir] & gf_mask_GPU(GF_ZID_WIDTH));
    fep_dev->fep_road[ie][ir] = (evt_dev->evt_road[ie][ir] & gf_mask_GPU(SVT_ROAD_WIDTH));
    fep_dev->fep_sect[ie][ir] = (evt_dev->evt_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH));
    fep_dev->fep_cable_sect[ie][ir] = (evt_dev->evt_cable_sect[ie][ir] & gf_mask_GPU(SVT_SECT_WIDTH));
    fep_dev->fep_err[ie][ir] = evt_dev->evt_err[ie][ir];
 
  }

  fep_dev->fep_nroads[ie]  = evt_dev->evt_nroads[ie];
  fep_dev->fep_ee_word[ie] = evt_dev->evt_ee_word[ie];
  fep_dev->fep_err_sum[ie] = evt_dev->evt_err_sum[ie];

}

__global__ void gf_fep_set_GPU (evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt) {

  int ie, ir, ic;
  int icomb; /* The number of combinations */

  ie = blockIdx.x; // events index
  ir = blockIdx.y; // roads index
  ic = threadIdx.x; // comb index

  // first initialize fep arrays
  fep_dev->fep_lcl[ie][ir][ic] = 0;
  fep_dev->fep_hitmap[ie][ir][ic] = 0;
  fep_dev->fep_phi[ie][ir][ic] = 0;
  fep_dev->fep_crv[ie][ir][ic] = 0;
  fep_dev->fep_lclforcut[ie][ir][ic] = 0;
  fep_dev->fep_ncomb5h[ie][ir][ic] = 0;
  fep_dev->fep_crv_sign[ie][ir][ic] = 0; 
  for (int id=0; id<XFT_LYR; id++) {
    fep_dev->fep_hit[ie][ir][ic][id] = 0;
    fep_dev->fep_hitZ[ie][ir][ic][id] = 0;
  }

  if ( ( ie < maxEvt ) && 
      ( ir < fep_dev->fep_nroads[ie] ) &&
      ( ic < fep_dev->fep_ncmb[ie][ir] ) ) {

    icomb = ic;

    for (int id=0; id<XFT_LYR; id++) {

      if (evt_dev->evt_nhits[ie][ir][id] != 0) {
        fep_dev->fep_hit[ie][ir][ic][id] = evt_dev->evt_hit[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]];
        fep_dev->fep_hitZ[ie][ir][ic][id] = evt_dev->evt_hitZ[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]];
        fep_dev->fep_lcl[ie][ir][ic] |= ((evt_dev->evt_lcl[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]] & gf_mask_GPU(1)) << id);
        fep_dev->fep_lclforcut[ie][ir][ic] |= ((evt_dev->evt_lclforcut[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]] & gf_mask_GPU(1)) << id);
        icomb /= evt_dev->evt_nhits[ie][ir][id];
        fep_dev->fep_hitmap[ie][ir][ic] |= (1<<id);
      } /* if (evt_dev->evt_nhits[ie][ir][id] |= 0)  */

    } /* for (id=0; id<XFT_LYR; id++) */

    /* check if this is a 5/5 track */
    if (fep_dev->fep_hitmap[ie][ir][ic] != 0x1f) 
      fep_dev->fep_ncomb5h[ie][ir][ic] = 1;
    else
      fep_dev->fep_ncomb5h[ie][ir][ic] = 5;
      
    if (evt_dev->evt_nhits[ie][ir][XFT_LYR] != 0) {
      fep_dev->fep_phi[ie][ir][ic] = (evt_dev->evt_phi[ie][ir][icomb%evt_dev->evt_nhits[ie][ir][XFT_LYR]] & gf_mask_GPU(SVT_PHI_WIDTH));
      fep_dev->fep_crv[ie][ir][ic] = (evt_dev->evt_crv[ie][ir][icomb%evt_dev->evt_nhits[ie][ir][XFT_LYR]] & gf_mask_GPU(SVT_CRV_WIDTH));
      fep_dev->fep_crv_sign[ie][ir][ic] = (evt_dev->evt_crv_sign[ie][ir][icomb%evt_dev->evt_nhits[ie][ir][XFT_LYR]]);
    }
  }
}

__global__ void gf_fit_format_GPU (struct fep_arrays* fep_dev, 
                                    struct fit_arrays* fit_dev, int maxEvt ) {

  int ie, ir, ic, ich;
  long long int temp = 0;

  ie = blockIdx.x; // events index
  ir = blockIdx.y; // roads index

  // combination indexes
  ic = threadIdx.x;
  ich = threadIdx.y;

  if ( ( ie < maxEvt ) && 
      ( ir < fep_dev->fep_nroads[ie] ) && 
      ( ic < fep_dev->fep_ncmb[ie][ir] ) && 
      ( ich < fep_dev->fep_ncomb5h[ie][ir][ic] ) ) {

    /* phi */
    temp = fit_dev->fit_fit[ie][0][ir][ic][ich];
    if ( temp > 0) {
      temp++;
      temp = temp >> 1;
    } else {
      temp--;
      temp = -((-temp) >> 1);
    }
    if (abs(temp) > gf_mask_GPU(OPHI_WIDTH)) {
      fit_dev->fit_err[ie][ir][ic][ich] |= (1<<FIT_RESULT_OFLOW_BIT);
    }

    temp = (temp < 0 ? -temp : temp) & gf_mask_GPU(OPHI_WIDTH);
    fit_dev->fit_fit[ie][0][ir][ic][ich] = temp;

    /* impact parameter */
    temp = fit_dev->fit_fit[ie][1][ir][ic][ich];
    if ( temp > 0) {
      temp++;
      temp = temp >> 1;
    } else {
      temp--;
      temp = -((-temp) >> 1);
    }
       /*overflow check */
    if (abs(temp) > gf_mask_GPU(OIMP_WIDTH)) {
      fit_dev->fit_err[ie][ir][ic][ich] |= (1<< FIT_RESULT_OFLOW_BIT);
    }

    temp = (temp < 0 ? -temp : temp) & gf_mask_GPU(OIMP_WIDTH);
    /* now add a bit for the sign  */
    if ( fit_dev->fit_fit[ie][1][ir][ic][ich] < 0) {
      temp += (1<<OIMP_SIGN);
    }
    fit_dev->fit_fit[ie][1][ir][ic][ich] = temp;

    /* curvature */
    temp = fit_dev->fit_fit[ie][2][ir][ic][ich];
    if (temp > 0) {
      temp++;
      temp = temp >> 1;
    } else {
      temp--;
      temp = -((-temp) >> 1);
    }
    /*overflow check */
    if (abs(temp) > gf_mask_GPU(OCVR_WIDTH)) {
      fit_dev->fit_err[ie][ir][ic][ich] |= (1<<FIT_RESULT_OFLOW_BIT);
    }
    temp = (temp < 0 ? -temp : temp) & gf_mask_GPU(OCVR_WIDTH);
    /*  now add a bit for the sign  */
    if (fit_dev->fit_fit[ie][2][ir][ic][ich] < 0) {
      temp += (1<<OCVR_SIGN);
    }
    fit_dev->fit_fit[ie][2][ir][ic][ich] = temp;

    /* chi 1,2,3 */

    /*
    for(ichi = 3; ichi < 6; ichi++) {
      temp = fit_fit[ie + ichi*NEVTS + ir*NEVTS*6 + ic*NEVTS*6*MAXROAD + ich*NEVTS*6*MAXROAD*MAXCOMB];
      fit_fit[ie][ichi][ir][ic][ich] = temp;
    }
    */

  } // end if


}

__global__ void kFit(struct fep_arrays* fep_dev, struct extra_data* edata_dev,
                     struct fit_arrays* fit_dev, int maxEvt) {

   int ir, ic, ip, ih, il;
   int hit[SVTNHITS];
   long long int coeff[NFITTER][SVTNHITS];
   int coe_addr, int_addr; /* Address for coefficients and intercept */
   int mka_addr; /* Address for MKADDR memory */
   long long int theintcp = 0;
   int sign_crv = 0;
   int which, lwhich;
   int iz;
   int ie;
   int newhitmap;

   int map[7][7] = {
     { 0, 1, 2, 3, -1, 4, 5 }, /* 01235 */
     { 0, 1, 2, -1, 3, 4, 5 }, /* 01245 */
     { 0, 1, -1, 2, 3, 4, 5 }, /* 01345 */
     { 0, -1, 1, 2, 3, 4, 5 }, /* 02345 */
     { -1, 0, 1, 2, 3, 4, 5 }, /* 12345 */
     { 0, 1, 2, 3, -1, 4, 5 }, /* (??) */
     { 0, 1, 2, 3, -1, 4, 5 }  /* (??) */
   };

  ie = blockIdx.x; // event index
  ir = blockIdx.y; // road index

  ic = threadIdx.x; // combination index
  ip = threadIdx.y; // fitter index

  fit_dev->fit_err_sum[ie] = fep_dev->fep_err_sum[ie];

  if ( ( ie < maxEvt ) && 
        ( ir < fep_dev->fep_nroads[ie] ) && 
        ( ic < fep_dev->fep_ncmb[ie][ir] ) ) {

    if ( fep_dev->fep_hitmap[ie][ir][ic] != 0x1f ) { 

      gf_mkaddr_GPU(edata_dev, fep_dev->fep_hitmap[ie][ir][ic], fep_dev->fep_lcl[ie][ir][ic], fep_dev->fep_zid[ie][ir],
                  &coe_addr, &int_addr, &mka_addr, fit_dev->fit_err_sum);
    
      int_addr = (int_addr<<OFF_SUBA_LSB) + fep_dev->fep_road[ie][ir];

      iz = fep_dev->fep_zid[ie][ir]&7;
      which = coe_addr/6; 
      lwhich = which;

      which = edata_dev->whichFit[iz][which];
    
      for (ih = 0; ih < SVTNHITS; ih++) {

        coeff[ip][ih] = map[lwhich][ih] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][ih]][iz][which]);
      
          if ( ih<NSVX_PLANE ) {
        
            hit[ih] = ((fep_dev->fep_hit[ie][ir][ic][ih] << 1) + 1) & gf_mask_GPU(15); 
       
          } else if (ih == HIT_PHI) {
        
            hit[ih] = fep_dev->fep_phi[ie][ir][ic];
            hit[ih] -= edata_dev->wedge[ie]*SVTSIM_XFTPHIBINS/SVTSIM_NWEDGE;
            hit[ih] = ((hit[ih] << 3) + (1 << 2)) & gf_mask_GPU(15);
  
          } else if (ih == HIT_CRV) {

            sign_crv = fep_dev->fep_crv_sign[ie][ir][ic];
            hit[ih] = ((fep_dev->fep_crv[ie][ir][ic] << 8) + (1 << 7)) & gf_mask_GPU(15);
        
          }

      } /* end for(ih = 0; ih < SVTNHITS; ih++) */

      theintcp = edata_dev->lfitparfcon[ip][6][iz][which] << 18;

      gf_fit_proc_GPU(hit, sign_crv, coeff[ip], theintcp, &(fit_dev->fit_fit[ie][ip][ir][ic][0]), &(fit_dev->fit_err[ie][ir][ic][0]));      

    } else { /* 5/5 track transformed in 5 4/5 tracks*/

      for (ih = 0; ih < NSVX_PLANE; ih++) {
        for (il = 0; il < NSVX_PLANE; il++) { /* one call to gf_fit_proc  for each ih value */
        /* let's calculate the new hitmap */
          if (il != ih) {
            switch (ih) {
              case 0 :     /*  11110 */
                newhitmap = 0x1e;
              break;
              case 1 :     /*  11101 */
                newhitmap = 0x1d;
              break;
              case 2 :     /*  11011 */
                newhitmap = 0x1b;
              break;
              case 3 :     /*  10111 */
                newhitmap = 0x17;
              break;
              case 4 :     /*  01111 */
                newhitmap = 0x0f;
              break;
            }

            gf_mkaddr_GPU(edata_dev, newhitmap, fep_dev->fep_lcl[ie][ir][ic], fep_dev->fep_zid[ie][ir],
                            &coe_addr, &int_addr, &mka_addr, fit_dev->fit_err_sum);

            if (ih == 0){
              iz = fep_dev->fep_hitZ[ie][ir][ic][1];;
            } else {
              iz = fep_dev->fep_zid[ie][ir]&7;
            }
            which = coe_addr/6;
            lwhich = which;
            which = edata_dev->whichFit[iz][which];

            coeff[ip][il] = map[lwhich][il] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][il]][iz][which]);
            hit[il] = ((fep_dev->fep_hit[ie][ir][ic][il] << 1) + 1) & gf_mask_GPU(15);

          } else { // il == ih
            hit[il] = 0 ;
            coeff[ip][il]= 1;
          }
        } /* end for(il = 0; il <  NSVX_PLANE; il++)  */

        hit[HIT_PHI] = fep_dev->fep_phi[ie][ir][ic];
        hit[HIT_PHI] -= edata_dev->wedge[ie]*SVTSIM_XFTPHIBINS/SVTSIM_NWEDGE;
        hit[HIT_PHI] = ((hit[HIT_PHI] << 3) + (1 << 2)) & gf_mask_GPU(15);

        coeff[ip][HIT_PHI] = map[lwhich][HIT_PHI] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][HIT_PHI]][iz][which]);

        sign_crv = fep_dev->fep_crv_sign[ie][ir][ic];
        hit[HIT_CRV] = ((fep_dev->fep_crv[ie][ir][ic] << 8) + (1 << 7)) & gf_mask_GPU(15);

        coeff[ip][HIT_CRV] = map[lwhich][HIT_CRV] < 0 ? 0 : (edata_dev->lfitparfcon[ip][map[lwhich][HIT_CRV]][iz][which]);

        /* INTERCEPT */
        theintcp = edata_dev->lfitparfcon[ip][6][iz][which] << 18;

        gf_fit_proc_GPU(hit, sign_crv, coeff[ip], theintcp, &(fit_dev->fit_fit[ie][ip][ir][ic][ih]), &(fit_dev->fit_err[ie][ir][ic][ih]));

        fit_dev->fit_err_sum[ie] |= fit_dev->fit_err[ie][ir][ic][ih];

      } /* end for(ih = 0; ih < NSVX_PLANE; ih++) */
    } /* end if(tf->fep_hitmap[ie][ir][ic] != 0x1f) */
  } /* enf if on indexes */
}

__global__ void gf_comparator_GPU(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, 
                                  struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int maxEvt) {

  int ie, ir, ic;
  int ChiSqCut, gvalue, gvalue_best;

  int ich = 0;
  int ind_best = 0;
  int chi2_best = 0;

  int gvalue_cut = 0x70;
  int bestTrackFound = 0;

  long long int chi[3], chi2;

  ie = blockIdx.x;
  ir = blockIdx.y;

  ic = threadIdx.x;

  if ( ( ie < maxEvt ) &&
        ( ir < fep_dev->fep_nroads[ie] ) &&
        ( ic < fep_dev->fep_ncmb[ie][ir] )) {

    ChiSqCut = 0x40;
    gvalue_best = 0x70;

    if (fep_dev->fep_ncomb5h[ie][ir][ic] == 1) {
      for (int i=0; i<NCHI; i++)
        chi[i] = fit_dev->fit_fit[ie][i+3][ir][ic][0];
      gf_chi2_GPU(chi, &fit_dev->fit_err[ie][ir][ic][0], &chi2);

      if (chi2 <= ChiSqCut) {
        chi2 = chi2 >> 2;
        gvalue = gf_gfunc_GPU(fep_dev->fep_ncomb5h[ie][ir][ic], ich, fep_dev->fep_hitmap[ie][ir][ic], fep_dev->fep_lcl[ie][ir][ic], (chi2 & gf_mask_GPU(CHI2SUM_WIDTH)));     
        if (gvalue < gvalue_cut) 
          gf_formatter_GPU(ie, ir, ic, 0, chi2, fep_dev, fit_dev, evt_dev, fout_dev);
      }
    } else if (fep_dev->fep_ncomb5h[ie][ir][ic] == 5) {
      bestTrackFound = 0;
      gvalue_best = 999;
      ind_best = 999;
      chi2_best = 999;
      for (ich = 0; ich < fep_dev->fep_ncomb5h[ie][ir][ic]; ich++) {
        for (int i=0; i<NCHI; i++) 
          chi[i] = fit_dev->fit_fit[ie][i+3][ir][ic][ich];
        /*  calculate chisq */
        gf_chi2_GPU(chi, &fit_dev->fit_err[ie][ir][ic][ich], &chi2);
        /* check chiSq  */
        if (chi2 <= ChiSqCut) {
          chi2 = chi2 >> 2; /* FC - hack .. see matching shift in gf_chi2 */
          gvalue = gf_gfunc_GPU(fep_dev->fep_ncomb5h[ie][ir][ic], ich, fep_dev->fep_hitmap[ie][ir][ic], fep_dev->fep_lcl[ie][ir][ic], (chi2 & gf_mask_GPU(CHI2SUM_WIDTH)));
          if  ((gvalue < gvalue_cut) && (gvalue < gvalue_best)) {
            gvalue_best = gvalue;
            ind_best = ich;
            chi2_best = chi2;
            bestTrackFound = 1;
          }
        } /*  end if(chi2 <= ChiSqCut) */
      } /*  end for(ich = 0; ich < gf->fep->ncomb5h[ir][ic]; ich++) */

      if (bestTrackFound) 
        gf_formatter_GPU(ie, ir, ic, ind_best, chi2_best, fep_dev, fit_dev, evt_dev, fout_dev);

    } /* end  if(gf->fep->ncomb5h[ir][ic] == 1) */

  } /* end if on indexes */

}

__global__ void gf_compute_eeword_GPU( struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, 
                                       struct fout_arrays* fout_dev, int maxEvt) {

  int   eoe_err;
  int   ie = blockIdx.x * blockDim.x + threadIdx.x;

  if ( ie < maxEvt ) {
    fout_dev->fout_err_sum[ie] = (fep_dev->fep_err_sum[ie] | fit_dev->fit_err_sum[ie]);
    gf_formatter_err_GPU(fout_dev->fout_err_sum[ie], GF_ERRMASK_CDF,
                      GF_ERRMASK_SVT, GF_ERRMASK_EOE,
                      &eoe_err, &fout_dev->fout_cdferr[ie],
                      &fout_dev->fout_svterr[ie]);

    fout_dev->fout_ee_word[ie] = (fep_dev->fep_ee_word[ie] &
                                (gf_mask_GPU(SVT_WORD_WIDTH) & ~(1<<SVT_PAR_BIT)));
    fout_dev->fout_ee_word[ie] |= (eoe_err<<SVT_ERR_LSB);
    fout_dev->fout_ee_word[ie] |= (fout_dev->fout_parity[ie]<<SVT_PAR_BIT); 
  } 
}


void gf_unpack_GPU(unsigned int *data_in, int n_words, struct evt_arrays *evt_dev, int *d_tEvts ) {

  thrust::device_vector<unsigned int> d_vec(n_words);
  thrust::copy(data_in, data_in + n_words, d_vec.begin());
  stop_time("input copy and initialize");

  start_time();
  thrust::device_vector<unsigned int> d_idt(n_words);
  thrust::device_vector<unsigned int> d_out1t(n_words);
  thrust::device_vector<unsigned int> d_out2t(n_words);
  thrust::device_vector<unsigned int> d_out3t(n_words);

  thrust::transform(
  thrust::make_zip_iterator(thrust::make_tuple(d_vec.begin(), d_vec.begin()-1)),
  thrust::make_zip_iterator(thrust::make_tuple(d_vec.end(), d_vec.end()-1)),
  thrust::make_zip_iterator(thrust::make_tuple(d_idt.begin(), d_out1t.begin(),
                 d_out2t.begin(), d_out3t.begin())),
                 unpacker());

  thrust::device_vector<unsigned int> d_evt(n_words);
  thrust::device_vector<unsigned int> d_road(n_words);
  thrust::device_vector<unsigned int> d_rhit(n_words);
  thrust::device_vector<unsigned int> d_lhit(n_words);

  thrust::exclusive_scan(
    thrust::make_transform_iterator(d_idt.begin(), isNewEvt()),
    thrust::make_transform_iterator(d_idt.end(),   isNewEvt()),
    d_evt.begin());
  thrust::exclusive_scan_by_key(
    d_evt.begin(), d_evt.end(), // keys
    thrust::make_transform_iterator(d_idt.begin(), isNewRoad()), // vals
    d_road.begin());
  thrust::inclusive_scan_by_key(
    d_road.begin(), d_road.end(), // keys
    thrust::make_transform_iterator(d_idt.begin(), isNewHit()), //vals
    d_rhit.begin());
  thrust::inclusive_scan_by_key(
    d_idt.begin(), d_idt.end(), // keys
    thrust::make_transform_iterator(d_idt.begin(), isNewHit()), //vals
    d_lhit.begin(),
    isEqualLayer()); // binary predicate

  thrust::device_vector<unsigned int> d_roadKey(n_words);
  thrust::device_vector<unsigned int> d_ncomb(n_words);
  thrust::pair<IntIterator, IntIterator> new_end;
  new_end = thrust::reduce_by_key(
              d_road.begin(), d_road.end(), // keys
              d_lhit.begin(), // vals
              d_roadKey.begin(), // keys output
              d_ncomb.begin(), // vals output
              thrust::equal_to<int>(), // binary predicate
              layerHitMultiply()); // binary operator

  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
        d_idt.begin(), d_idt.begin()+1, d_out1t.begin(), d_out2t.begin(), d_out3t.begin(),
        d_evt.begin(), d_road.begin(), d_rhit.begin(), d_lhit.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
        d_idt.end(), d_idt.end()+1, d_out1t.end(), d_out2t.end(), d_out3t.end(),
        d_evt.end(), d_road.end(), d_rhit.end(), d_lhit.end())),
      fill_tf_gpu(evt_dev, d_tEvts));

}

void gf_fit_GPU(tf_arrays_t tf, unsigned int *data_in, int n_words) {

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
 
//  printf("Fitting %d events...\n", tEvts);
 
  // Fep comb and set
  start_time();  
  gf_fep_comb_GPU<<<NEVTS, MAXROAD>>>(evt_dev, fep_dev, tEvts);
  gf_fep_set_GPU<<<blocks, MAXCOMB>>>(evt_dev, fep_dev, tEvts);
  stop_time("compute fep combinations");
  
  // Fit and set Fout
  start_time();
  kFit<<<blocks, dim3(MAXCOMB,NFITTER)>>>(fep_dev, edata_dev, fit_dev, tEvts);
//  stop_time("kFit");
//  start_time();
  gf_fit_format_GPU<<<blocks, dim3(MAXCOMB, MAXCOMB5H)>>>(fep_dev, fit_dev, tEvts);
//  stop_time("fit_format");
//  start_time();
  gf_comparator_GPU<<<blocks, dim3(MAXCOMB)>>>(fep_dev, evt_dev, fit_dev, fout_dev, tEvts);
  gf_compute_eeword_GPU<<<(NEVTS+255)/256, 256>>>(fep_dev, fit_dev, fout_dev, tEvts); 
  stop_time("fit data and set output");

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
  unsigned int data_send[2500000];
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

  if ( strcmp(where,"cpu") == 0 ) {
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
  } else {
    printf("Start work on GPU..... \n");
    gf_fit_GPU(tf, data_send, k);
    printf(".... fits %d events! \n", tf->totEvts);
    // build "cable" output structure
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
