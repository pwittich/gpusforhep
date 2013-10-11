#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#include "svt_utils_opencl.h"

__constant int gf_maskdata_GPU[] = {
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

__constant unsigned long gf_maskdata3_GPU[] = {
  0x000000000000UL,
  0x000000000001UL, 0x000000000003UL, 0x000000000007UL, 0x00000000000fUL,
  0x00000000001fUL, 0x00000000003fUL, 0x00000000007fUL, 0x0000000000ffUL,
  0x0000000001ffUL, 0x0000000003ffUL, 0x0000000007ffUL, 0x000000000fffUL,
  0x000000001fffUL, 0x000000003fffUL, 0x000000007fffUL, 0x00000000ffffUL,
  0x00000001ffffUL, 0x00000003ffffUL, 0x00000007ffffUL, 0x0000000fffffUL,
  0x0000001fffffUL, 0x0000003fffffUL, 0x0000007fffffUL, 0x000000ffffffUL,
  0x000001ffffffUL, 0x000003ffffffUL, 0x000007ffffffUL, 0x00000fffffffUL,
  0x00001fffffffUL, 0x00003fffffffUL, 0x00007fffffffUL, 0x0000ffffffffUL,
  0x0001ffffffffUL, 0x0003ffffffffUL, 0x0007ffffffffUL, 0x000fffffffffUL,
  0x001fffffffffUL, 0x003fffffffffUL, 0x007fffffffffUL, 0x00ffffffffffUL,
  0x01ffffffffffUL, 0x03ffffffffffUL, 0x07ffffffffffUL, 0x0fffffffffffUL,
  0x1fffffffffffUL, 0x3fffffffffffUL, 0x7fffffffffffUL, 0xffffffffffffUL 
};

__kernel void gf_fep_comb_GPU (global struct evt_arrays* evt_dev, global struct fep_arrays* fep_dev) {

  int ie, ir;

  int nlyr; /* The number of layers with a hit */
  int ncomb; /* The number of combinations */

  ie = get_group_id(0); // events index
  ir = get_local_id(0); // roads index

  fep_dev->fep_ncmb[ie][ir] = 0;
  fep_dev->fep_zid[ie][ir] = 0;
  fep_dev->fep_road[ie][ir] = 0;
  fep_dev->fep_sect[ie][ir] = 0;
  fep_dev->fep_cable_sect[ie][ir] = 0;
  fep_dev->fep_err[ie][ir] = 0;
  //fep_dev->fep_err[ie][ir] = evt_dev->evt_nhits[ie][ir][0];

  if ( ( ie < evt_dev->totEvts ) &&
      ( ir < evt_dev->evt_nroads[ie] ) ) {

    ncomb = 1;
    nlyr = 0;
    /* At first, we calculate how many combinations are there */
    for (int id=0; id<(XFT_LYR+1); id++) {
      if (evt_dev->evt_nhits[ie][ir][id] != 0) {
        ncomb = ncomb*evt_dev->evt_nhits[ie][ir][id];
        nlyr++;
      }
    }

    if ( nlyr < MINHITS )
      evt_dev->evt_err[ie][ir] |= (1<<UFLOW_HIT_BIT);

    fep_dev->fep_ncmb[ie][ir] = ncomb;
    atom_or(&evt_dev->evt_err_sum[ie], evt_dev->evt_err[ie][ir]);

    fep_dev->fep_zid[ie][ir] = (evt_dev->evt_zid[ie][ir] & gf_maskdata_GPU[GF_ZID_WIDTH]);
    fep_dev->fep_road[ie][ir] = (evt_dev->evt_road[ie][ir] & gf_maskdata_GPU[SVT_ROAD_WIDTH]);
    fep_dev->fep_sect[ie][ir] = (evt_dev->evt_sect[ie][ir] & gf_maskdata_GPU[SVT_SECT_WIDTH]);
    fep_dev->fep_cable_sect[ie][ir] = (evt_dev->evt_cable_sect[ie][ir] & gf_maskdata_GPU[SVT_SECT_WIDTH]);
    //fep_dev->fep_err[ie][ir] = evt_dev->evt_err[ie][ir];

  }

  fep_dev->fep_nroads[ie]  = evt_dev->evt_nroads[ie];
  fep_dev->fep_ee_word[ie] = evt_dev->evt_ee_word[ie];
  fep_dev->fep_err_sum[ie] = evt_dev->evt_err_sum[ie];

}

__kernel void gf_fep_set_GPU (global struct evt_arrays* evt_dev, global struct fep_arrays* fep_dev) {

  int ie, ir, ic;
  int icomb; /* The number of combinations */

  ie = get_group_id(0); // events index
  ir = get_group_id(1); // roads index
  ic = get_local_id(0); // comb index

  

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
  
  if ( ( ie < evt_dev->totEvts ) &&
       ( ir < fep_dev->fep_nroads[ie] ) &&
       ( ic < fep_dev->fep_ncmb[ie][ir] ) ) {
    
    icomb = ic;
    
    for (int id=0; id<XFT_LYR; id++) {
      
      if (evt_dev->evt_nhits[ie][ir][id] != 0) {
        fep_dev->fep_hit[ie][ir][ic][id] = evt_dev->evt_hit[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]];
        fep_dev->fep_hitZ[ie][ir][ic][id] = evt_dev->evt_hitZ[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]];
        fep_dev->fep_lcl[ie][ir][ic] |= ((evt_dev->evt_lcl[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]] & gf_maskdata_GPU[1]) << id);
        fep_dev->fep_lclforcut[ie][ir][ic] |= ((evt_dev->evt_lclforcut[ie][ir][id][icomb%evt_dev->evt_nhits[ie][ir][id]] & gf_maskdata_GPU[1]) << id);
        icomb /= evt_dev->evt_nhits[ie][ir][id];
        fep_dev->fep_hitmap[ie][ir][ic] |= (1<<id);
      } // if (evt_dev->evt_nhits[ie][ir][id] |= 0) 
      
    } // for (id=0; id<XFT_LYR; id++) 
    
      // check if this is a 5/5 track
    if (fep_dev->fep_hitmap[ie][ir][ic] != 0x1f)
      fep_dev->fep_ncomb5h[ie][ir][ic] = 1;
    else
      fep_dev->fep_ncomb5h[ie][ir][ic] = 5;
    
    if (evt_dev->evt_nhits[ie][ir][XFT_LYR] != 0) {
      fep_dev->fep_phi[ie][ir][ic] = (evt_dev->evt_phi[ie][ir][icomb%evt_dev->evt_nhits[ie][ir][XFT_LYR]] & gf_maskdata_GPU[SVT_PHI_WIDTH]);
      fep_dev->fep_crv[ie][ir][ic] = (evt_dev->evt_crv[ie][ir][icomb%evt_dev->evt_nhits[ie][ir][XFT_LYR]] & gf_maskdata_GPU[SVT_CRV_WIDTH]);
      fep_dev->fep_crv_sign[ie][ir][ic] = (evt_dev->evt_crv_sign[ie][ir][icomb%evt_dev->evt_nhits[ie][ir][XFT_LYR]]);
    }
    
  }
  
  
}






