#include "svtsim_defines.h"
#include "svt_utils_opencl.h"

extern "C" {

  int gf_init(tf_arrays_t* ptr_tf);
  int gf_init_evt(evt_arrays_t* ptr_evt);
  //int svtsim_fconread(tf_arrays_t tf);
  int svtsim_fconread(tf_arrays_t tf, struct extra_data* edata);

  int gf_fep_unpack(tf_arrays_t tf, int n_words_in, void* data);
  int gf_fep_unpack_evt(evt_arrays_t tf, int n_words_in, void* data);
  int gf_fep_comb(tf_arrays_t tf);
  int gf_comparator(tf_arrays_t tf);
  int gf_fit(tf_arrays_t tf);
  void svtsim_cable_addwords(svtsim_cable_t *cable, unsigned int *word, int nword);
  void svtsim_cable_addword(svtsim_cable_t *cable, unsigned int word);
  void svtsim_cable_copywords(svtsim_cable_t *cable, unsigned int *word, int nword);
  svtsim_cable_t * svtsim_cable_new(void);

  void gf_unpack_GPU(unsigned int *data_in, int n_words, struct evt_arrays *evt_dev, int *d_tEvts );
  void gf_fep_GPU( evt_arrays* evt_dev, fep_arrays* fep_dev, int maxEvt );
  void gf_fit_GPU(struct fep_arrays* fep_dev, struct evt_arrays* evt_dev, struct extra_data* edata_dev,
                struct fit_arrays* fit_dev, struct fout_arrays* fout_dev, int maxEvt);

}


