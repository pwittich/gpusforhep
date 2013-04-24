#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable


	//constData[6][7]
	// 0 - D0
	// 1 - PT
	// 2 - PHI
	// 3 - CHI0
	// 4 - CHI1
	// 5 - CHI2

    __constant DATAWORD constData[6*7]   =
	     {0x5963, 0x7f80, 0x14b5, 0xdf54, 0xe3e2, 0x2e73, 0x4317, //D0
	     0x41e9, 0x62bf, 0xaecf, 0x75a2, 0x127d, 0xdb7f, 0x0a6e,  //PT
	     0x6963, 0x6f80, 0x24b5, 0xcf54, 0xf3e2, 0x1e73, 0x5317,  //PHI
	     0x42e9, 0x61bf, 0xafcf, 0x74a2, 0x137d, 0xda7f, 0x0b6e,  //Chi0
	     0x5973, 0x7f70, 0x14c5, 0xdf44, 0xe3f2, 0x2e63, 0x4327,  //Chi1
	     0x41ea, 0x62be, 0xaec0, 0x75a1, 0x127e, 0xdb7e, 0x0a6f}; //Chi2



__kernel void
kTrigger_CopyOnly(
	__global DATAWORD *data_in,
	__global DATAWORD *data_out)
{

    long thr_idx = get_local_id(0);
    long idx = get_global_id(0);

    data_out[4 * idx] = data_in[thr_idx];
    data_out[4 * idx + 1] = data_in[thr_idx];
    data_out[4 * idx + 2] = data_in[thr_idx];
    data_out[4 * idx + 3] = data_in[thr_idx];

}

__kernel void
kTrigger_big(
	__global DATAWORD *data_in,
	__global DATAWORD *data_out)
{

    long thr_idx = get_local_id(0);
    long idx = get_global_id(0);
    long blk_idx = get_group_id(0);


    DATAWORD h[7] = {1,0,0,0,0,0,0};
    DATAWORD res[6] = {0,0,0,0,0,0};
    DATAWORD chi2 = 0;

    unsigned int i = 1;
    unsigned int j = 0;

    //calculate h
    for(;i<7;++i) 
    {
	h[i] = data_in[thr_idx];
	h[i] &= 0x0000ffff << blk_idx;
	h[i] >>= 4*(i-1);
    }
	   
    //calculate results
    for(i=0;i<6;++i) 
	for(j=0;j<7;++j) 
	    res[i] += h[j]*constData[i*7+j];

    data_out[4 * idx] = res[0];//h[0];
    data_out[4 * idx + 1] = data_in[thr_idx];
    data_out[4 * idx + 2] = data_in[thr_idx];
    data_out[4 * idx + 3] = data_in[thr_idx];

    //assemble chi2
    for(i=3;i<6;++i) 
        chi2 += res[i]*res[i];
    
    //save to output memory
    for(i=0;i<3;++i) 
    	data_out[4*idx + i] = res[i];
    data_out[4*idx + 3] = chi2;
}

__kernel void
kTrigger_SP(
	__global DATAWORD *data_in,
	__global DATAWORD *data_out)
{

    long thr_idx = get_local_id(0);
    long idx = get_global_id(0);
    long blk_idx = get_group_id(0);

    DATAWORD h[7] = {1,0,0,0,0,0,0};
    DATAWORD res[6] = {0,0,0,0,0,0};
    DATAWORD chi2 = 0;

    unsigned int i = 1;
    unsigned int j = 0;

    //calculate h
    for(;i<7;++i) 
    {
	h[i] = data_in[thr_idx];
	h[i] &= 0x0000ffff << blk_idx;
	h[i] >>= 4*(i-1);
    }	

    //calculate results
    for(i=0;i<6;++i) 
	for(j=0;j<7;++j) 
		res[i] += h[j]*constData[i*7+j];

    //assemble chi2
    for(i=3;i<6;++i) 
        chi2 += res[i]*res[i];


    //long outIdx = 0;
    //outIdx = atomic_add(data_in, 1); //ngood?? atomicAdd(ngood, 1);

    // passes track requirements so calculate track parameters and output

    //save to output memory
    for(i=0;i<3;++i) 
    	data_out[4*idx + i] = res[i];
    data_out[4*idx + 3] = chi2;
    	//data_out[4*outIdx + i] = res[i];
    //data_out[4*outIdx + 3] = chi2;
}


/////////////////////////////////
/////////////////////////////////
/////////////////////////////////
/*

__global__ void
kTrigger_SP(unsigned int *ngood, unsigned int *data_out, unsigned int *data_in)
{

    long blk_idx = blockIdx.x;
    long thr_idx = threadIdx.x;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long outIdx = 0;

    unsigned int c, chi2;
    unsigned int h0 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx));
    unsigned int h1 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 1);
    unsigned int h2 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 2);
    unsigned int h3 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 3);
    unsigned int h4 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 4);
    unsigned int h5 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 5);

    // first calculate chi2
    c = constCHI0[0] 
	+ h0*constCHI0[1] + h1*constCHI0[2] + h2*constCHI0[3] 
	+ h3*constCHI0[4] + h4*constCHI0[5] + h5*constCHI0[6];
    chi2  = c*c;
    c = constCHI1[0] 
	+ h0*constCHI1[1] + h1*constCHI1[2] + h2*constCHI1[3] 
	+ h3*constCHI1[4] + h4*constCHI1[5] + h5*constCHI1[6];
    chi2 += c*c;
    c = constCHI2[0] 
	+ h0*constCHI2[1] + h1*constCHI2[2] + h2*constCHI2[3] 
	+ h3*constCHI2[4] + h4*constCHI2[5] + h5*constCHI2[6];
    chi2 += c*c;

    //if (rand() / RAND_MAX < 1.0/1000) {
    //if (idx < 10) {
    //if (chi2 > 500000) return;
    if (chi2 > 50000000) return;

    // passes track requirements so calculate track parameters and output

    outIdx = atomicAdd(ngood, 1);
    //outIdx = *ngood - 1;

    data_out[4*outIdx]   = constD0[0]  
	+ h0*constD0[1]  + h1*constD0[2]  + h2*constD0[3]  
	+ h3*constD0[4]  + h4*constD0[5]  + h5*constD0[6];
    data_out[4*outIdx+1] = constPT[0]  
	+ h0*constPT[1]  + h1*constPT[2]  + h2*constPT[3]  
	+ h3*constPT[4]  + h4*constPT[5]  + h5*constPT[6];
    data_out[4*outIdx+2] = constPHI[0] 
	+ h0*constPHI[1] + h1*constPHI[2] + h2*constPHI[3] 
	+ h3*constPHI[4] + h4*constPHI[5] + h5*constPHI[6];
    data_out[4*outIdx+3] = chi2;

    // data_out[4*outIdx] = data_out[4*outIdx+1] = data_out[4*outIdx+2] = data_out[4*outIdx+3] = outIdx;

}

__global__ void
kTrigger_big(char *flag, unsigned int *data_out, unsigned int *data_in)
{

    //unsigned int constD0[7]   =
    //{0x5963, 0x7f80, 0x14b5, 0xdf54, 0xe3e2, 0x2e73, 0x4317};
    //unsigned int constPT[7]   =
    //{0x41e9, 0x62bf, 0xaecf, 0x75a2, 0x127d, 0xdb7f, 0x0a6e};
    //unsigned int constPHI[7]  =
    //{0x6963, 0x6f80, 0x24b5, 0xcf54, 0xf3e2, 0x1e73, 0x5317};
    //unsigned int constCHI0[7] =
    //{0x42e9, 0x61bf, 0xafcf, 0x74a2, 0x137d, 0xda7f, 0x0b6e};
    //unsigned int constCHI1[7] =
    //{0x5973, 0x7f70, 0x14c5, 0xdf44, 0xe3f2, 0x2e63, 0x4327};
    //unsigned int constCHI2[7] =
    //{0x41ea, 0x62be, 0xaec0, 0x75a1, 0x127e, 0xdb7e, 0x0a6f};

    long blk_idx = blockIdx.x;
    long thr_idx = threadIdx.x;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int c, chi2;
    unsigned int h0 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx));
    unsigned int h1 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 1);
    unsigned int h2 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 2);
    unsigned int h3 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 3);
    unsigned int h4 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 4);
    unsigned int h5 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4 * 5);

    //data_out[4*idx]=data_in[idx];
    //data_out[4*idx+1]=data_in[idx];
    //data_out[4*idx+2]=data_in[idx];
    //data_out[4*idx+3]=data_in[idx];

    data_out[4*idx]  = constD0[0]  
	+ h0*constD0[1]  + h1*constD0[2]  + h2*constD0[3]  
	+ h3*constD0[4]  + h4*constD0[5]  + h5*constD0[6];
    data_out[4*idx+1]  = constPT[0]  
	+ h0*constPT[1]  + h1*constPT[2]  + h2*constPT[3]  
	+ h3*constPT[4]  + h4*constPT[5]  + h5*constPT[6];
    data_out[4*idx+2] = constPHI[0] 
	+ h0*constPHI[1] + h1*constPHI[2] + h2*constPHI[3] 
	+ h3*constPHI[4] + h4*constPHI[5] + h5*constPHI[6];
    c = constCHI0[0] 
	+ h0*constCHI0[1] + h1*constCHI0[2] + h2*constCHI0[3] 
	+ h3*constCHI0[4] + h4*constCHI0[5] + h5*constCHI0[6];
    chi2  = c*c;
    c = constCHI1[0] 
	+ h0*constCHI1[1] + h1*constCHI1[2] + h2*constCHI1[3] 
	+ h3*constCHI1[4] + h4*constCHI1[5] + h5*constCHI1[6];
    chi2 += c*c;
    c = constCHI2[0] 
	+ h0*constCHI2[1] + h1*constCHI2[2] + h2*constCHI2[3] 
	+ h3*constCHI2[4] + h4*constCHI2[5] + h5*constCHI2[6];
    data_out[4*idx+3] = chi2 + c*c;

}

__global__ void
kTrigger(char *flag, unsigned int *data_out, unsigned int *data_in)
{

    //unsigned int constD0[7]   =
    //{0x5963, 0x7f80, 0x14b5, 0xdf54, 0xe3e2, 0x2e73, 0x4317};
    //unsigned int constPT[7]   =
    //{0x41e9, 0x62bf, 0xaecf, 0x75a2, 0x127d, 0xdb7f, 0x0a6e};
    //unsigned int constPHI[7]  =
    //{0x6963, 0x6f80, 0x24b5, 0xcf54, 0xf3e2, 0x1e73, 0x5317};
    //unsigned int constCHI0[7] =
    //{0x42e9, 0x61bf, 0xafcf, 0x74a2, 0x137d, 0xda7f, 0x0b6e};
    //unsigned int constCHI1[7] =
    //{0x5973, 0x7f70, 0x14c5, 0xdf44, 0xe3f2, 0x2e63, 0x4327};
    //unsigned int constCHI2[7] =
    //{0x41ea, 0x62be, 0xaec0, 0x75a1, 0x127e, 0xdb7e, 0x0a6f};

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c, chi2;
    unsigned int h0 = ((data_in[idx]) & 0x0000ffff);
    unsigned int h1 = ((data_in[idx]) & 0x0000ffff) >> (4 * 1);
    unsigned int h2 = ((data_in[idx]) & 0x0000ffff) >> (4 * 2);
    unsigned int h3 = ((data_in[idx]) & 0x0000ffff) >> (4 * 3);
    unsigned int h4 = ((data_in[idx]) & 0x0000ffff) >> (4 * 4);
    unsigned int h5 = ((data_in[idx]) & 0x0000ffff) >> (4 * 5);

    //data_out[4*idx]=data_in[idx];
    //data_out[4*idx+1]=data_in[idx];
    //data_out[4*idx+2]=data_in[idx];
    //data_out[4*idx+3]=data_in[idx];

    data_out[4*idx]  = constD0[0]  
	+ h0*constD0[1]  + h1*constD0[2]  + h2*constD0[3]  
	+ h3*constD0[4]  + h4*constD0[5]  + h5*constD0[6];
    data_out[4*idx+1]  = constPT[0]  
	+ h0*constPT[1]  + h1*constPT[2]  + h2*constPT[3]  
	+ h3*constPT[4]  + h4*constPT[5]  + h5*constPT[6];
    data_out[4*idx+2] = constPHI[0] 
	+ h0*constPHI[1] + h1*constPHI[2] + h2*constPHI[3] 
	+ h3*constPHI[4] + h4*constPHI[5] + h5*constPHI[6];
    c = constCHI0[0] 
	+ h0*constCHI0[1] + h1*constCHI0[2] + h2*constCHI0[3] 
	+ h3*constCHI0[4] + h4*constCHI0[5] + h5*constCHI0[6];
    chi2  = c*c;
    c = constCHI1[0] 
	+ h0*constCHI1[1] + h1*constCHI1[2] + h2*constCHI1[3] 
	+ h3*constCHI1[4] + h4*constCHI1[5] + h5*constCHI1[6];
    chi2 += c*c;
    c = constCHI2[0] 
	+ h0*constCHI2[1] + h1*constCHI2[2] + h2*constCHI2[3] 
	+ h3*constCHI2[4] + h4*constCHI2[5] + h5*constCHI2[6];
    data_out[4*idx+3] = chi2 + c*c;

}

*/