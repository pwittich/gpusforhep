#include "svt_utils.h"


__global__ void
k_word_decode(int N, unsigned int *words, int *ids, int *out1, int *out2, int *out3) {

  /* 
    parallel word_decode kernel.
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

  int word = words[idx];
  int ee, ep, lyr;

  lyr = -999; /* Any invalid numbers != 0-7 */

  out1[idx] = 0;
  out2[idx] = 0;
  out3[idx] = 0;

  if (word > gf_mask_GPU(SVT_WORD_WIDTH)) {
    ids[idx] = lyr;
    return;
  }

  /* check if this is a EP or EE word */
  ee = (word >> SVT_EE_BIT)  & gf_mask_GPU(1);
  ep = (word >> SVT_EP_BIT)  & gf_mask_GPU(1);

  int prev_word = (idx==0) ? 0 : words[idx-1];
  int p_ee = (prev_word >> SVT_EE_BIT) & gf_mask_GPU(1);
  int p_ep = (prev_word >> SVT_EP_BIT) & gf_mask_GPU(1);

  // check if this is the second XFT word
//  bool xft = ((prev_word >> SVT_LYR_LSB) & gf_mask_GPU(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;
  bool xft = !p_ee && !p_ep && ((prev_word >> SVT_LYR_LSB) & gf_mask_GPU(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;


  if (ee && ep) { /* End of Event word */
    out1[idx] = word; // ee_word
    lyr = EE_LYR;
  } else if (ee) { /* only EE bit ON is error condition */
    lyr = EE_LYR; /* We have to check */
  } else if (ep) { /* End of Packet word */
    lyr = EP_LYR;
    out1[idx] = 6; // sector
    out2[idx] = word  & gf_mask_GPU(AMROAD_WORD_WIDTH); // amroad
  } else if (xft) { /* Second XFT word */
    out1[idx] = (word >> SVT_CRV_LSB)  & gf_mask_GPU(SVT_CRV_WIDTH); // crv
    out2[idx] = (word >> (SVT_CRV_LSB + SVT_CRV_WIDTH))  & gf_mask_GPU(1); // crv_sign
    out3[idx] = word & gf_mask_GPU(SVT_PHI_WIDTH); // phi
    lyr = XFT_LYR_2;
  } else { /* SVX hits or the first XFT word */
    lyr = (word >> SVT_LYR_LSB)  & gf_mask_GPU(SVT_LYR_WIDTH);
    if (lyr == XFT_LYRID) lyr = XFT_LYR; // probably don't need - stp
    out1[idx] = (word >> SVT_Z_LSB)  & gf_mask_GPU(SVT_Z_WIDTH); // zid
    out2[idx] = (word >> SVT_LCLS_BIT) & gf_mask_GPU(1); // lcl
    out3[idx] = word & gf_mask_GPU(SVT_HIT_WIDTH); // hit
  }

  ids[idx] = lyr;
}

void gf_unpack_cuda_GPU(unsigned int *d_data_in, int n_words, struct evt_arrays *evt_dev, int* d_tEvts ) {

  int N_THREADS_PER_BLOCK = 32;
  int *ids, *out1, *out2, *out3;
  int *d_ids, *d_out1, *d_out2, *d_out3;
  int tEvts = 0;
 // unsigned int *d_data_in;
  long sizeW = sizeof(int) * n_words;
  
  ids  = (int *)malloc(sizeW);
  out1 = (int *)malloc(sizeW);
  out2 = (int *)malloc(sizeW);
  out3 = (int *)malloc(sizeW);

  cudaMalloc((void **)&d_ids, sizeW);
  cudaMalloc((void **)&d_out1, sizeW);
  cudaMalloc((void **)&d_out2, sizeW);
  cudaMalloc((void **)&d_out3, sizeW);
 // cudaMalloc((void **)&d_data_in, sizeW);

  // Copy data to the Device
 // cudaMemcpy(d_data_in, data_in, sizeW, cudaMemcpyHostToDevice);
  
  k_word_decode <<<(n_words+N_THREADS_PER_BLOCK-1)/N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>
  (n_words, d_data_in, d_ids, d_out1, d_out2, d_out3);

  cudaMemcpy(ids, d_ids, sizeW, cudaMemcpyDeviceToHost);
  cudaMemcpy(out1, d_out1, sizeW, cudaMemcpyDeviceToHost);
  cudaMemcpy(out2, d_out2, sizeW, cudaMemcpyDeviceToHost);
  cudaMemcpy(out3, d_out3, sizeW, cudaMemcpyDeviceToHost);

  ///////////////// now fill evt (gf_fep_unpack)

  struct evt_arrays* evta = (evt_arrays*)malloc(sizeof(struct evt_arrays));

  memset(evta->evt_nroads, 0, sizeof(evta->evt_nroads));
  memset(evta->evt_err_sum, 0, sizeof(evta->evt_err_sum));
  memset(evta->evt_layerZ, 0, sizeof(evta->evt_layerZ));
  memset(evta->evt_nhits, 0,  sizeof(evta->evt_nhits));
  memset(evta->evt_err,  0,   sizeof(evta->evt_err));
  memset(evta->evt_zid,  0,   sizeof(evta->evt_zid));

  for (int ie = 0; ie < NEVTS; ie++) {
    evta->evt_zid[ie][evta->evt_nroads[ie]] = -1; // because we set it to 0 for GPU version
  }


  int id_last = -1;
  int evt = EVT;
  int id;

  for (int i = 0; i < n_words; i++) {
        
    id = ids[i];

    bool gf_xft = 0;
    if (id == XFT_LYR_2) { // compatibility - stp
      id = XFT_LYR;
      gf_xft = 1;
    }

    int nroads = evta->evt_nroads[evt];
    int nhits = evta->evt_nhits[evt][nroads][id];

    // SVX Data
    if (id < XFT_LYR) {
      int zid = out1[i];
      int lcl = out2[i];
      int hit = out3[i];

      evta->evt_hit[evt][nroads][id][nhits] = hit;
      evta->evt_hitZ[evt][nroads][id][nhits] = zid;
      evta->evt_lcl[evt][nroads][id][nhits] = lcl;
      evta->evt_lclforcut[evt][nroads][id][nhits] = lcl;
      evta->evt_layerZ[evt][nroads][id] = zid;

      if (evta->evt_zid[evt][nroads] == -1) {
        evta->evt_zid[evt][nroads] = zid & gf_mask(GF_SUBZ_WIDTH);
      } else {
        evta->evt_zid[evt][nroads] = (((zid & gf_mask(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH)
                                + (evta->evt_zid[evt][nroads] & gf_mask(GF_SUBZ_WIDTH)));
      }

      nhits = ++evta->evt_nhits[evt][nroads][id];

      // Error Checking
      if (nhits == MAX_HIT) evta->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
      if (id < id_last) evta->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);
    } else if (id == XFT_LYR && gf_xft == 0) {
      // we ignore - stp
    } else if (id == XFT_LYR && gf_xft == 1) {
      int crv = out1[i];
      int crv_sign = out2[i];
      int phi = out3[i];

      evta->evt_crv[evt][nroads][nhits] = crv;
      evta->evt_crv_sign[evt][nroads][nhits] = crv_sign;
      evta->evt_phi[evt][nroads][nhits] = phi;

      nhits = ++evta->evt_nhits[evt][nroads][id];

      // Error Checking
      if (nhits == MAX_HIT) evta->evt_err[evt][nroads] |= (1 << OFLOW_HIT_BIT);
      if (id < id_last) evta->evt_err[evt][nroads] |= (1 << OUTORDER_BIT);
    } else if (id == EP_LYR) {
      int sector = out1[i];
      int amroad = out2[i];

      evta->evt_cable_sect[evt][nroads] = sector;
      evta->evt_sect[evt][nroads] = sector;
      evta->evt_road[evt][nroads] = amroad;
      evta->evt_err_sum[evt] |= evta->evt_err[evt][nroads];

      nroads = ++evta->evt_nroads[evt];

      if (nroads > MAXROAD) {
        printf("The limit on the number of roads fitted by the TF is %d\n",MAXROAD);
        printf("You reached that limit evt->nroads = %d\n",nroads);
      }

      for (id = 0; id <= XFT_LYR; id++)
        evta->evt_nhits[evt][nroads][id] = 0;

      evta->evt_err[evt][nroads] = 0;
      evta->evt_zid[evt][nroads] = -1;

      id = -1; id_last = -1;
    } else if (id == EE_LYR) {

      evta->evt_ee_word[evt] = out1[i];
      tEvts++;
      evt++;

      id = -1; id_last = -1;
    } else {
      printf("Error INV_DATA_BIT: layer = %u\n", id);
      evta->evt_err[evt][nroads] |= (1 << INV_DATA_BIT);
    }
    id_last = id;

  } //end loop on input words

  free(ids);
  free(out1);
  free(out2);
  free(out3);

 // cudaFree(d_data_in);
  cudaFree(d_ids);
  cudaFree(d_out1);
  cudaFree(d_out2);
  cudaFree(d_out3);


  cudaMemcpy(evt_dev, evta, sizeof(struct evt_arrays), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tEvts, &tEvts, sizeof(int), cudaMemcpyHostToDevice);

  free(evta);
}

