#include "svt_utils.h"

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

void gf_fep_GPU( evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt ) {

  dim3 blocks(NEVTS,MAXROAD);

  gf_fep_comb_GPU<<<NEVTS, MAXROAD>>>(evt_dev, fep_dev, maxEvt);
  gf_fep_set_GPU<<<blocks, MAXCOMB>>>(evt_dev, fep_dev, maxEvt);

}


