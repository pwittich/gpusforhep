#ifndef SVTSIM_MIC
#define SVTSIM_MIC

#pragma offload_attribute(push, target(mic))

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include "svt_utils.h"

extern "C" {
  int gf_init(tf_arrays_t* ptr_tf);
  int svtsim_fconread(tf_arrays_t tf, struct extra_data* edata);

}
namespace mic {

inline void empty() {
}

inline void destroy(evt_arrays* evt_dev, fep_arrays* fep_dev, fit_arrays* fit_dev, fout_arrays* fout_dev)
{
  delete evt_dev;
  delete fep_dev;
  delete fit_dev;
  delete fout_dev;
  
}

// unpack section

inline void init_evt(evt_arrays*& evt_dev, fep_arrays*& fep_dev)
{
  evt_dev = new evt_arrays;

  memset(evt_dev->evt_nroads,  0, sizeof(evt_dev->evt_nroads));
  memset(evt_dev->evt_err_sum, 0, sizeof(evt_dev->evt_err_sum));
  memset(evt_dev->evt_layerZ,  0, sizeof(evt_dev->evt_layerZ));
  memset(evt_dev->evt_nhits,   0, sizeof(evt_dev->evt_nhits));
  memset(evt_dev->evt_err,     0, sizeof(evt_dev->evt_err));
  memset(evt_dev->evt_zid,     0, sizeof(evt_dev->evt_zid));

  fep_dev = new fep_arrays;

}

typedef thrust::tuple<unsigned int, unsigned int>     DataTuple;

typedef thrust::tuple<unsigned int, unsigned int,
		      unsigned int, unsigned int>     UnpackTuple;

typedef thrust::host_vector<unsigned int>::iterator IntIterator;
typedef thrust::tuple<IntIterator, IntIterator,
		      IntIterator, IntIterator>       IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple>           ZipIterator;

struct unpacker : public thrust::unary_function<DataTuple, UnpackTuple>
{
  UnpackTuple operator()(DataTuple t)
  {
    unsigned int word = thrust::get<0>(t);
    unsigned int prev_word = thrust::get<1>(t);
    unsigned int val1 = 0, val2 = 0, val3 = 0;

    int ee, ep, lyr;

    lyr = -999; /* Any invalid numbers != 0-7 */

    /* check if this is a EP or EE word */
    ee = (word >> SVT_EE_BIT)  & gf_mask_GPU(1);
    ep = (word >> SVT_EP_BIT)  & gf_mask_GPU(1);

    int p_ee = (prev_word >> SVT_EE_BIT) & gf_mask_GPU(1);
    int p_ep = (prev_word >> SVT_EP_BIT) & gf_mask_GPU(1);

    // check if this is the second XFT word
    bool xft = !p_ee && !p_ep && ((prev_word >> SVT_LYR_LSB) & gf_mask_GPU(SVT_LYR_WIDTH)) == XFT_LYR ? 1 : 0;

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

struct isNewRoad: public thrust::unary_function<unsigned int, bool>
{
  bool operator()(unsigned int id) const
  {
    return id == EP_LYR;
  }
};

struct isNewHit: public thrust::unary_function<unsigned int, bool>
{
  bool operator()(unsigned int id) const
  {
    return id < XFT_LYR || id == XFT_LYR_2;
  }
};

struct isNewEvt: public thrust::unary_function<unsigned int, bool>
{
  bool operator()(unsigned int id) const {
    return id == EE_LYR;
  }
};

struct isEqualLayer: public thrust::binary_function<unsigned int, unsigned int, bool>
{
  bool operator()(unsigned int a, unsigned int b) const {
    return a == b || ((a == XFT_LYR || a == XFT_LYR_2) && (b == XFT_LYR || b == XFT_LYR_2));
  }
};

struct fill_tf
{
  int* totEvts_;
  evt_arrays* evt_dev_;
  fill_tf(evt_arrays* evt_dev, int* totEvts) : evt_dev_(evt_dev), totEvts_(totEvts)
  {
    *totEvts_ = 0;
  }

  template <typename Tuple>
  void operator()(Tuple const& t) {
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

      evt_dev_->evt_hit[evt][road][id][lhit] = hit;
      evt_dev_->evt_hitZ[evt][road][id][lhit] = zid;
      evt_dev_->evt_lcl[evt][road][id][lhit] = lcl;
      evt_dev_->evt_lclforcut[evt][road][id][lhit] = lcl;
      evt_dev_->evt_layerZ[evt][road][id] = zid;

      if (rhit == 0) {
        atomicOr(&evt_dev_->evt_zid[evt][road], zid & gf_mask_GPU(GF_SUBZ_WIDTH));
      } else if (id_next == XFT_LYR) {
        atomicOr(&evt_dev_->evt_zid[evt][road], (zid & gf_mask_GPU(GF_SUBZ_WIDTH)) << GF_SUBZ_WIDTH);
      }

      atomicAdd(&evt_dev_->evt_nhits[evt][road][id], 1);

      // Error Checking
      if (lhit == MAX_HIT)
        evt_dev_->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);

    } else if (id == XFT_LYR) {
      // we ignore but leave here to not trigger 'else' case - stp
    } else if (id == XFT_LYR_2) {
      id = XFT_LYR; // for XFT_LYR_2 kludge - stp
      int crv      = out1;
      int crv_sign = out2;
      int phi      = out3;

      evt_dev_->evt_crv[evt][road][lhit] = crv;
      evt_dev_->evt_crv_sign[evt][road][lhit] = crv_sign;
      evt_dev_->evt_phi[evt][road][lhit] = phi;

      atomicAdd(&evt_dev_->evt_nhits[evt][road][id], 1);

      // Error Checking
      if (lhit == MAX_HIT)
        evt_dev_->evt_err[evt][road] |= (1 << OFLOW_HIT_BIT);
    } else if (id == EP_LYR) {
      int sector = out1;
      int amroad = out2;

      evt_dev_->evt_cable_sect[evt][road] = sector;
      evt_dev_->evt_sect[evt][road] = sector;
      evt_dev_->evt_road[evt][road] = amroad;
      evt_dev_->evt_err_sum[evt] |= evt_dev_->evt_err[evt][road];

      atomicAdd(&evt_dev_->evt_nroads[evt], 1);

    } else if (id == EE_LYR) {
      int ee_word = out1;

      evt_dev_->evt_ee_word[evt] = ee_word;
      atomicAdd(totEvts_, 1);
    } else {
      evt_dev_->evt_err[evt][road] |= (1 << INV_DATA_BIT);
    }
  }
};


inline void gf_unpack(unsigned int const* data_in, int n_words, evt_arrays* evt_dev, int& totEvts)
{
  thrust::host_vector<unsigned int> d_vec(n_words+1);
  d_vec[0] = 0;
  thrust::copy(data_in, data_in + n_words, d_vec.begin()+1);

  thrust::host_vector<unsigned int> d_idt(n_words);
  thrust::host_vector<unsigned int> d_out1t(n_words);
  thrust::host_vector<unsigned int> d_out2t(n_words);
  thrust::host_vector<unsigned int> d_out3t(n_words);

  thrust::transform(
    thrust::make_zip_iterator(thrust::make_tuple(d_vec.begin()+1, d_vec.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(d_vec.end(), d_vec.end()-1)),
    thrust::make_zip_iterator(thrust::make_tuple(d_idt.begin(), d_out1t.begin(),
                                                 d_out2t.begin(), d_out3t.begin())),
    unpacker());

  thrust::host_vector<unsigned int> d_evt(n_words);
  thrust::host_vector<unsigned int> d_road(n_words);
  thrust::host_vector<unsigned int> d_rhit(n_words);
  thrust::host_vector<unsigned int> d_lhit(n_words);

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

  thrust::for_each(
    thrust::make_zip_iterator(
      thrust::make_tuple(
        d_idt.begin(), d_idt.begin()+1, d_out1t.begin(), d_out2t.begin(), d_out3t.begin(),
        d_evt.begin(), d_road.begin(), d_rhit.begin(), d_lhit.begin())
    ),
    thrust::make_zip_iterator(
      thrust::make_tuple(
        d_idt.end(), d_idt.end()+1, d_out1t.end(), d_out2t.end(), d_out3t.end(),
        d_evt.end(), d_road.end(), d_rhit.end(), d_lhit.end())
    ),
    fill_tf(evt_dev, &totEvts));
}

// fep section

inline void init_fep(fep_arrays*& fep_dev, int ie)
{

  memset(fep_dev->fep_lcl[ie],  0, MAXROAD*MAXCOMB*sizeof(int));
  memset(fep_dev->fep_hitmap[ie],  0, MAXROAD*MAXCOMB*sizeof(int));
  memset(fep_dev->fep_phi[ie],  0, MAXROAD*MAXCOMB*sizeof(int));
  memset(fep_dev->fep_crv[ie],  0, MAXROAD*MAXCOMB*sizeof(int));
  memset(fep_dev->fep_lclforcut[ie],  0, MAXROAD*MAXCOMB*sizeof(int));
  memset(fep_dev->fep_hit[ie],  0, MAXROAD*MAXCOMB*NSVX_PLANE*sizeof(int));
  memset(fep_dev->fep_hitZ[ie],  0, MAXROAD*MAXCOMB*NSVX_PLANE*sizeof(int));
}

inline void gf_fep_comb_Mic (evt_arrays* evt_dev, fep_arrays* fep_dev, int ie) {

  int nlyr; /* The number of layers with a hit */
  int ncomb, icomb; /* The number of combinations */

  for (int ir = 0; ir < evt_dev->evt_nroads[ie]; ir++) { /* Loop for the number of roads */

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

    for (int ic = 0; ic < fep_dev->fep_ncmb[ie][ir]; ic++) { /* Loop for the number of combinations */

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

  fep_dev->fep_nroads[ie]  = evt_dev->evt_nroads[ie];
  fep_dev->fep_ee_word[ie] = evt_dev->evt_ee_word[ie];
  fep_dev->fep_err_sum[ie] = evt_dev->evt_err_sum[ie];

}

// fit section

inline void init_fit(fit_arrays*& fit_dev, fout_arrays*& fout_dev)
{

  fit_dev = new fit_arrays;
  fout_dev = new fout_arrays;

  memset(fout_dev->fout_ntrks, 0, sizeof(fout_dev->fout_ntrks));
  memset(fout_dev->fout_parity, 0, sizeof(fout_dev->fout_parity));
  memset(fout_dev->fout_ee_word, 0, sizeof(fout_dev->fout_ee_word));
  memset(fout_dev->fout_err_sum, 0,  sizeof(fout_dev->fout_err_sum));
  memset(fout_dev->fout_cdferr,  0,   sizeof(fout_dev->fout_cdferr));
  memset(fout_dev->fout_svterr,  0,   sizeof(fout_dev->fout_svterr));
}

int svtsim_whichFit_full_GPU(int layerMask, int lcMask) {

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

int svtsim_whichFit_GPU(struct extra_data* edata_dev, int zin, int layerMask, int lcMask) {

   int which0 = 0, which = 0;
   if (zin<0 || zin>=SVTSIM_NBAR) zin = 0;
   which0 = svtsim_whichFit_full_GPU(layerMask, lcMask);
   which = edata_dev->whichFit[zin][which0];

   return which;
}


int  svtsim_get_gfMkAddr_GPU(struct extra_data* edata_dev, int *d, int nd, int d0) {

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

int gf_mkaddr_GPU(struct extra_data* edata_dev, int hitmap, int lclmap, int zmap,
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

int gf_fit_proc_GPU(int hit[], int sign_crv, long long int coeff[], 
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


int gf_chi2_GPU(long long int chi[], int* trk_err, long long int *chi2) {

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

  if ((err>>OFLOW_HIT_BIT)&gf_mask_GPU(1))
    word |= (1<<GFS_OFL_HIT);

  if ((err>>OFLOW_CHI_BIT)&gf_mask_GPU(1))
    word |= (1<<GFS_OFL_CHI);

  if (((err>>UFLOW_HIT_BIT)&gf_mask_GPU(1)) ||
       ((err>>OUTORDER_BIT)&gf_mask_GPU(1)))
    word |= (1<<GFS_INV_DATA);

  return word;

}

int cal_parity_GPU(int word) {

  int par = 0;

  for (int i=0; i<SVT_WORD_WIDTH; i++)
    par ^= ((word>>i) & gf_mask_GPU(1));

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

int gf_formatter_GPU(int ie, int ir, int ic, int ich, int chi2, 
                            struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, struct evt_arrays* evt_dev,
                            struct fout_arrays* fout_dev) {

  int it, err;
  int hit_form[NSVX_PLANE];

  int z = 0; /* z should be 6 bits large */
  int gf_stat = 0;

  // atomicAdd (on mic) returns the *new* value
  it = atomicAdd(&fout_dev->fout_ntrks[ie], 1);
  it--; 
 
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

inline void gf_fit_format_Mic (struct fep_arrays* fep_dev, 
                                    struct fit_arrays* fit_dev, int ie ) {

  long long int temp = 0;

  for (int ir = 0; ir < fep_dev->fep_nroads[ie]; ir++ ) { 
    for (int ic = 0; ic < fep_dev->fep_ncmb[ie][ir]; ic++ ) { 
      for (int ich = 0; ich < fep_dev->fep_ncomb5h[ie][ir][ic]; ich++ ) { 

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

      } // end for ich
    } // end for ic
  } // end for ir


}

inline void kFit(struct fep_arrays* fep_dev, struct extra_data* edata_dev,
                     struct fit_arrays* fit_dev, int ie) {

   int ip, ih, il;
   int hit[SVTNHITS];
   long long int coeff[NFITTER][SVTNHITS];
   int coe_addr, int_addr; /* Address for coefficients and intercept */
   int mka_addr; /* Address for MKADDR memory */
   long long int theintcp = 0;
   int sign_crv = 0;
   int which, lwhich;
   int iz;
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


  fit_dev->fit_err_sum[ie] = fep_dev->fep_err_sum[ie];

  for (int ir=0; ir < fep_dev->fep_nroads[ie]; ir++) { /* Loop for all roads */
    for (int ic=0; ic < fep_dev->fep_ncmb[ie][ir]; ic++) { /* Loop for all combinations */
    
    if ( fep_dev->fep_hitmap[ie][ir][ic] != 0x1f ) { 

      gf_mkaddr_GPU(edata_dev, fep_dev->fep_hitmap[ie][ir][ic], fep_dev->fep_lcl[ie][ir][ic], fep_dev->fep_zid[ie][ir],
                  &coe_addr, &int_addr, &mka_addr, fit_dev->fit_err_sum);
    
      int_addr = (int_addr<<OFF_SUBA_LSB) + fep_dev->fep_road[ie][ir];

      iz = fep_dev->fep_zid[ie][ir]&7;
      which = coe_addr/6; 
      lwhich = which;

      which = edata_dev->whichFit[iz][which];
   
 
      for (ip=0; ip<NFITTER; ip++) {
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

      }/* end for (ip=0; ip<NFITTER; ip++)  */
    } else { /* 5/5 track transformed in 5 4/5 tracks*/
     for (ip=0; ip<NFITTER; ip++) {
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
      }/* end for (ip=0; ip<NFITTER; ip++)  */
    } /* end if(tf->fep_hitmap[ie][ir][ic] != 0x1f) */
  } /* end ic */
  } // end ir
}

inline void gf_comparator_Mic(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, 
                                  struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int ie) {

  int ChiSqCut, gvalue, gvalue_best;

  int ich = 0;
  int ind_best = 0;
  int chi2_best = 0;

  int gvalue_cut = 0x70;
  int bestTrackFound = 0;
  int eoe_err;

  long long int chi[3], chi2;

  for (int ir=0; ir < fep_dev->fep_nroads[ie]; ir++) { /* Loop for all roads */
   for (int ic=0; ic < fep_dev->fep_ncmb[ie][ir]; ic++) { /* Loop for all combinations */

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

    } // end  if(gf->fep->ncomb5h[ir][ic] == 1) 
   } // end for on ic
  } // end for on ir

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

inline void gf_compute_eeword_Mic( struct fep_arrays* fep_dev, struct fit_arrays* fit_dev, 
                                       struct fout_arrays* fout_dev, int ie) {

  int   eoe_err;

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


// Output section

 /*
  * Wrap memory allocation to allow debug, etc.
  */
 
 typedef struct memDebug_s {
   struct memDebug_s *last, *next;
   const char *fileName;
   int lineNum, userBytes;
   int deadBeef;
 } memDebug_t;
 static int n_alloc = 0, t_alloc = 0, m_alloc = 0;
 static memDebug_t *chain_alloc = 0;
  static char svtsim_err_str[10][256];
static int svtsim_n_err;


 
 int svtsim_memDebug(int chainDump);
 void *svtsim_malloc1(size_t size, const char *filename, int linenum);
 
 void svtsim_free1(void *tmp, const char *filename, int linenum);
 void *svtsim_realloc1(void *inptr, size_t size, 
 		      const char *filename, int linenum);
 #define svtsim_malloc(x) (svtsim_malloc1((x), __FILE__, __LINE__))
 #define svtsim_free(x) (svtsim_free1((x), __FILE__, __LINE__))
 #define svtsim_mallocForever(x) (svtsim_malloc1((x), __FILE__, -__LINE__))
 #define svtsim_realloc(x, y) (svtsim_realloc1((x), (y), __FILE__, __LINE__))
 
 void svtsim_assert_set(char *filename, int line)
 {
   if(svtsim_n_err<10) sprintf(svtsim_err_str[svtsim_n_err],"SVTSIM::ASSERT %s: %d\n",filename,line);
   svtsim_n_err++;
 }
 
 #define svtsim_assert(x) \
   do { \
   if ((x)) continue; \
   svtsim_assert_set( __FILE__, __LINE__ ); \
   } while (0)
 
 
 
 int
 svtsim_memDebug(int nChainDump)
 {
   int i = 0, t = 0, tb = 0;
   memDebug_t *p = chain_alloc;
   fprintf(stderr, "svtsim_memDebug: n_alloc=%d t_alloc=%d chain+1=%p\n", 
 	  n_alloc, t_alloc, chain_alloc+1);
   for (; p; p = p->next, i++) {
     svtsim_assert(p->deadBeef==0xdeadBeef);
     svtsim_assert(p->next==0 || p->next->last==p);
     t++;
     tb += p->userBytes;
     if (p->lineNum<0) continue; 
     if (nChainDump<0 || i<nChainDump)
       fprintf(stderr, "  p=%p sz=%d line=%s:%d\n", 
 	      p+1, p->userBytes, p->fileName, p->lineNum);
   }
   svtsim_assert(t==n_alloc);
   svtsim_assert(tb==t_alloc);
   return t_alloc;
 }
 
 
 void *
 svtsim_malloc1(size_t size, const char *filename, int linenum)
 {
   memDebug_t *p = 0;
   p = (memDebug_t *) malloc(size+sizeof(*p));
   if(p == NULL) printf("FAILED MALLOC %s %d!\n",filename,linenum);
   svtsim_assert(p!=NULL);
   p->deadBeef = 0xdeadBeef;
   p->userBytes = size;
   p->fileName = filename;
   p->lineNum = linenum;
   p->last = 0;
   p->next = chain_alloc;
   chain_alloc = p;
   if (p->next) {
     svtsim_assert(p->next->last==0);
     p->next->last = p;
   }
   memset(p+1, 0, size);
   n_alloc++;
   t_alloc += p->userBytes;
   if (t_alloc>m_alloc) m_alloc = t_alloc;
 #if 0
   for (i = 0; i<SVTSIM_NEL(badaddr); i++) 
     if (p+1==badaddr[i]) {
       svtsim_memDebug(2);
       svtsim_assert(0);
     }
 #endif
   return((void *) (p+1));
 }
 
 void 
 svtsim_free1(void *p, const char *filename, int linenum)
 {
   int nbytes = 0;
   memDebug_t *q = ((memDebug_t *)p)-1;
   if (!p) return;
   svtsim_assert(p!=(void *)0xffffffff);
   if (q->deadBeef!=0xdeadBeef) {
     fprintf(stderr, "%p->deadBeef==0x%x (%s:%d)\n", 
 	    q, q->deadBeef, filename, linenum);
     free(p);
     return;
   }
   svtsim_assert(q->deadBeef==0xdeadBeef);
   svtsim_assert(q->lineNum>=0); 
   if (q->last) {
     svtsim_assert(q->last->next==q);
     q->last->next = q->next; 
   } else {
     svtsim_assert(chain_alloc==q);
     chain_alloc = q->next;
   }
   if (q->next) {
     svtsim_assert(q->next->last==q);
     q->next->last = q->last;
   }
   n_alloc--;
   t_alloc -= q->userBytes;
   nbytes = sizeof(*q)+q->userBytes;
   memset(q, -1, nbytes);
   free(q);
 }
 
 void *
 svtsim_realloc1(void *inptr, size_t size, const char *filename, int linenum)
 {
   memDebug_t *p = ((memDebug_t *) inptr)-1;
   if (!inptr) return svtsim_malloc1(size, filename, linenum);
   if (!size) { svtsim_free1(inptr, filename, linenum); return 0; }
   if (0) { 
     svtsim_free1(inptr, filename, linenum);
     return svtsim_malloc1(size, filename, linenum);
   }
   svtsim_assert(p->deadBeef==0xdeadBeef);
   if (p->last) {
     svtsim_assert(p->last->next==p);
   } else {
     svtsim_assert(p==chain_alloc);
   }
   if (p->next) svtsim_assert(p->next->last==p);
   t_alloc -= p->userBytes;
   p = (memDebug_t *) realloc(p, size+sizeof(*p));
   if (p->last) p->last->next = p; else chain_alloc = p;
   if (p->next) p->next->last = p;
   p->userBytes = size;
   t_alloc += p->userBytes;
   if (t_alloc>m_alloc) m_alloc = t_alloc;
   svtsim_assert(p!=0);
 #if 0
   for (i = 0; i<SVTSIM_NEL(badaddr); i++) 
     if (p+1==badaddr[i]) {
       svtsim_memDebug(2);
       svtsim_assert(0);
     }
 #endif
   return p+1;
 }
 
 
svtsim_cable_t * svtsim_cable_new(void) {
  svtsim_cable_t *cable = 0;
  cable = (svtsim_cable_t *)svtsim_malloc(sizeof(*cable));
  cable->data = 0; 
  cable->ndata = 0; 
  cable->mdata = 0;
  return cable; 
}
 

 
void svtsim_cable_addwords(svtsim_cable_t *cable, unsigned int  *word, int nword) {

  const int minwords = 8;
  int nnew = 0;
  svtsim_assert(cable); 
  nnew = cable->ndata + nword;
  if (nnew > cable->mdata) {
    cable->mdata = SVTSIM_MAX(minwords, 2*nnew);
    cable->data = (unsigned int *)svtsim_realloc(cable->data,cable->mdata*sizeof(cable->data[0]));
  }
  if (word) {
    memcpy(cable->data+cable->ndata, word, nword*sizeof(word[0]));
  } else {
    memset(cable->data+cable->ndata, 0, nword*sizeof(word[0]));
  }
  cable->ndata += nword;

  //printf("in svtsim_cable_addwords: cable->ndata = %d\n", cable->ndata);
}


void svtsim_cable_addword(svtsim_cable_t *cable, unsigned int word){
  svtsim_cable_addwords(cable, &word, 1);
}

void svtsim_cable_copywords(svtsim_cable_t *cable, unsigned int *word, int nword) {
  svtsim_assert(cable);
  cable->ndata = 0;
  svtsim_cable_addwords(cable, word, nword);
}


inline void set_outcable(fout_arrays* fout_dev, int totEvts, unsigned int *&data_rec, int &ow) {

  svtsim_cable_t *out;
  out = svtsim_cable_new();

  svtsim_cable_copywords(out, 0, 0);

  for (int ie=0; ie < totEvts; ie++) {
    for (int nt=0; nt < fout_dev->fout_ntrks[ie]; nt++) {
      svtsim_cable_addwords(out, fout_dev->fout_gfword[ie][nt], NTFWORDS);
    }
    svtsim_cable_addword(out, fout_dev->fout_ee_word[ie]);
  }

  ow = out->ndata;

  for (int i=0; i < ow ; i++) {
    data_rec[i] = out->data[i];
  }

  svtsim_free(out);
}


} // namespace mic

#pragma offload_attribute(pop)

#endif
