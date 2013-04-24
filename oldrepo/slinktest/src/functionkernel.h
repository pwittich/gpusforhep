#pragma once

#include <stddef.h>
#include "svtsim_functions.h"
//#pragma once is a non-standard but widely supported preprocessor directive designed 
//to cause the current source file to be included only once in a single compilation

#ifdef __cplusplus
extern "C" {
#endif

    void GPU_Init(int n_words);
    void GPU_Destroy();
  
    void launchTrackfitKernel(unsigned int *data_res, unsigned int *data_in);
    void launchSimpleKernel(unsigned int *data_res, unsigned int *data_in, size_t nwords);
    void launch4SimpleKernel(unsigned int *data_res, unsigned int *data_in);
    void CPU_Trigger(unsigned int *data_in, unsigned int *data_out, int nwords);

    void launchFepUnpackKernel(tf_arrays_t tf, unsigned int *data_in, int n_words);
    
    void launchFepCombKernel(unsigned int *data_res, unsigned int *data_in);
    void launchFitKernel(int *fit_fit_dev, int *fep_ncmb_dev);
    void launchTestKernel(int *dev_a, int *dev_b, int *dev_c);

    void launchTestKernel_tf(tf_arrays_t tf, unsigned int *data_in, int n_words);

#ifdef __cplusplus
}
#endif
