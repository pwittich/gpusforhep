#ifndef SVT_UTILS
#define SVT_UTILS

#include "svtsim_functions.h"

#define gf_mask_GPU(x) (gf_maskdata_GPU[(x)])
static int gf_maskdata_GPU[] = {
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
static long long int gf_maskdata3_GPU[] = {
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


struct evt_arrays {

  int evt_hit[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_hitZ[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_lcl[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_lclforcut[NEVTS][MAXROAD][NSVX_PLANE][MAX_HIT];
  int evt_layerZ[NEVTS][MAXROAD][NSVX_PLANE];
  int evt_zid[NEVTS][MAXROAD];
  int evt_nhits[NEVTS][MAXROAD][NSVX_PLANE+1];
  int evt_err[NEVTS][MAXROAD];
  int evt_crv[NEVTS][MAXROAD][MAX_HIT];
  int evt_crv_sign[NEVTS][MAXROAD][MAX_HIT];
  int evt_phi[NEVTS][MAXROAD][MAX_HIT];
  int evt_err_sum[NEVTS];
  int evt_cable_sect[NEVTS][MAXROAD];
  int evt_sect[NEVTS][MAXROAD];
  int evt_nroads[NEVTS];
  int evt_road[NEVTS][MAXROAD];
  int evt_ee_word[NEVTS];

};


 struct fep_arrays {

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
 //  int fep_err[NEVTS][MAXROAD][MAXCOMB][MAXCOMB5H];
   int fep_err[NEVTS][MAXROAD];
   int fep_crv_sign[NEVTS][MAXROAD][MAXCOMB];
   int fep_ncomb5h[NEVTS][MAXROAD][MAXCOMB];
   int fep_hitZ[NEVTS][MAXROAD][MAXCOMB][NSVX_PLANE];
   int fep_nroads[NEVTS];
   int fep_ee_word[NEVTS];
   int fep_err_sum[NEVTS];

 };

 struct extra_data {

   int wedge[NEVTS];
   int whichFit[SVTSIM_NBAR][FITBLOCK];/* handles TF mkaddr degeneracy */
   long long int lfitparfcon[NFITPAR][DIMSPA+1][SVTSIM_NBAR][FITBLOCK]; /* full-precision coeffs, P0s as read from fcon files SA*/

 };

 struct fit_arrays {

   long long int fit_fit[NEVTS][6][MAXROAD][MAXCOMB][MAXCOMB5H];
   int fit_err[NEVTS][MAXROAD][MAXCOMB][MAXCOMB5H];
   int fit_err_sum[NEVTS];

 };

 struct fout_arrays {

   int fout_parity[NEVTS];
   int fout_ntrks[NEVTS];
 //  int fout_iroad[NEVTS][MAXROAD*MAXCOMB];
 //  int fout_icmb[NEVTS][MAXROAD*MAXCOMB];
   unsigned int fout_gfword[NEVTS][MAXROAD*MAXCOMB][NTFWORDS];
   int fout_cdferr[NEVTS];
   int fout_svterr[NEVTS];
   int fout_ee_word[NEVTS];
   int fout_err_sum[NEVTS];

 };

#define atomicOr  __sync_or_and_fetch
#define atomicXor  __sync_xor_and_fetch
#define atomicAdd __sync_add_and_fetch

#endif
