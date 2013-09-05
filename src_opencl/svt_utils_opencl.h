#include "svtsim_defines.h"

#define MAX(x,y) ((x)>(y) ? (x):(y))

#define TIMING

typedef struct evt_arrays      *evt_arrays_t;

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
  unsigned long lfitparfcon[NFITPAR][DIMSPA+1][SVTSIM_NBAR][FITBLOCK]; /* full-precision coeffs, P0s as read from fcon files SA*/

};

struct fit_arrays {

  unsigned long fit_fit[NEVTS][6][MAXROAD][MAXCOMB][MAXCOMB5H];
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

