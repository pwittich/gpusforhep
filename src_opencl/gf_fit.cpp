#include "svt_utils_opencl.h"

int svtsim_whichFit_full_GPU(int layerMask, int lcMask) {
  return 0;
}

int svtsim_whichFit_GPU(struct extra_data* edata_dev, int zin, int layerMask, int lcMask) {
   return 0;
}


int  svtsim_get_gfMkAddr_GPU(struct extra_data* edata_dev, int *d, int nd, int d0) {
   return nd;
}

int gf_mkaddr_GPU(struct extra_data* edata_dev, int hitmap, int lclmap, int zmap,
                          int *coe_addr, int *int_addr, int *addr, int *err) {

  return 0;

}

int gf_fit_proc_GPU(int hit[], int sign_crv, long long int coeff[], 
                            long long int intcp, long long int *result, int *err) {
  return 0;
}


int gf_chi2_GPU(long long int chi[], int* trk_err, long long int *chi2) {
  return 0;

}

int gf_getq_GPU(int lyr_config) {

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

int gf_gfunc_GPU(int ncomb5h, int icomb5h, int hitmap, int lcmap, int chi2) {

  int lyr_config;
  int gvalue = 0;
  int newhitmap;
  int newlcmap;
  int q = 0;

  return gvalue;
}

int gf_stword_GPU(int id, int err) {
     /*
       Compose the GF status word in the 7th word from the GF 
       INPUT : err; error summary
       OUTPUT : return the gf_stword

       NOTE: Currently this code does not support the parity error and
             FIFO error.
     */

  int word;

  word = id;

  return word;

}

int cal_parity_GPU(int word) {

  int par = 0;

  return par;
}

int gf_formatter_err_GPU(int err, int cdfmsk, int svtmsk, int eoemsk,
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

  return 0;


}


int gf_formatter_GPU(int ie, int ir, int ic, int ich, int chi2, 
                            struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, struct evt_arrays* evt_dev,
                            struct fout_arrays* fout_dev) {

  return 0;

}

void gf_fit_format_GPU (struct fep_arrays* fep_dev, 
                                    struct fit_arrays* fit_dev, int maxEvt ) {


}

void kFit(struct fep_arrays* fep_dev, struct extra_data* edata_dev,
                     struct fit_arrays* fit_dev, int maxEvt) {
}

void gf_comparator_GPU(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, 
                                  struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int maxEvt) {
}

void gf_compute_eeword_GPU( struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, 
                                       struct fout_arrays* fout_dev, int maxEvt) {
}

void gf_fit_GPU(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, struct extra_data* edata_dev,
                struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int maxEvt) {

} 
