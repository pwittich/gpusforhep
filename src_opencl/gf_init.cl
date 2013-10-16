#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#include "svt_utils_opencl.h"

__kernel void init_arrays_GPU (global struct fout_arrays* fout_dev, global struct evt_arrays* evt_dev) {

  int ie, ir, ip;

  ie = get_group_id(0); // events index
  ie = get_group_id(1); // events index
  ip = get_local_id(0); // NSVX_PLANE+1 index

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
