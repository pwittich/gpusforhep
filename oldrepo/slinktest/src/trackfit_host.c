const unsigned int hconstD0[7]   =	{0x5963, 0x7f80, 0x14b5, 0xdf54, 
				 0xe3e2, 0x2e73, 0x4317};
const unsigned int hconstPT[7]   =	{0x41e9, 0x62bf, 0xaecf, 0x75a2, 
				 0x127d, 0xdb7f, 0x0a6e};
const unsigned int hconstPHI[7]  =	{0x6963, 0x6f80, 0x24b5, 0xcf54, 
				 0xf3e2, 0x1e73, 0x5317};
const unsigned int hconstCHI0[7] =	{0x42e9, 0x61bf, 0xafcf, 0x74a2, 
				 0x137d, 0xda7f, 0x0b6e};
const unsigned int hconstCHI1[7] =	{0x5973, 0x7f70, 0x14c5, 0xdf44, 
				 0xe3f2, 0x2e63, 0x4327};
const unsigned int hconstCHI2[7] =	{0x41ea, 0x62be, 0xaec0, 0x75a1, 
				 0x127e, 0xdb7e, 0x0a6f};

void CPU_Trigger_CopyOnly(unsigned int *data_in, unsigned int *data_out, int nwords, int thr_per_blk)
{

    int idx, thr_idx, blk_idx;
    for (idx=0; idx<nwords; idx++) {

	thr_idx = idx % thr_per_blk;
	blk_idx = idx / thr_per_blk;

	data_out[4*idx]=data_in[thr_idx];
	data_out[4*idx+1]=data_in[thr_idx];
	data_out[4*idx+2]=data_in[thr_idx];
	data_out[4*idx+3]=data_in[thr_idx];
    }

}


void CPU_Trigger_big(unsigned int *data_in, unsigned int *data_out, int nwords, int thr_per_blk)
{

    //printf("First word in (CPU): %x\n",data_in[0]);
    int idx, thr_idx, blk_idx;
    for (idx=0; idx<nwords; idx++) {

	thr_idx = idx % thr_per_blk;
	blk_idx = idx / thr_per_blk;


	unsigned int c, chi2;
	unsigned int h0 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx));
	unsigned int h1 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4*1);
	unsigned int h2 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4*2);
	unsigned int h3 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4*3);
	unsigned int h4 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4*4);
	unsigned int h5 = ((data_in[thr_idx]) & (0x0000ffff << blk_idx)) >> (4*5);

	//printf("Thread index %d, Block index %d, Data %x\n",thr_idx,blk_idx,h0);

	*(data_out+4*idx)   = hconstD0[0]  
	    + h0*hconstD0[1]  + h1*hconstD0[2]  + h2*hconstD0[3]  
	    + h3*hconstD0[4]  + h4*hconstD0[5]  + h5*hconstD0[6];
	*(data_out+4*idx+1) = hconstPT[0]  
	    + h0*hconstPT[1]  + h1*hconstPT[2]  + h2*hconstPT[3]  
	    + h3*hconstPT[4]  + h4*hconstPT[5]  + h5*hconstPT[6];
	*(data_out+4*idx+2) = hconstPHI[0] 
	    + h0*hconstPHI[1] + h1*hconstPHI[2] + h2*hconstPHI[3] 
	    + h3*hconstPHI[4] + h4*hconstPHI[5] + h5*hconstPHI[6];
	c = hconstCHI0[0] 
	    + h0*hconstCHI0[1] + h1*hconstCHI0[2] + h2*hconstCHI0[3] 
	    + h3*hconstCHI0[4] + h4*hconstCHI0[5] + h5*hconstCHI0[6];
	chi2  = c*c;
	c = hconstCHI1[0] 
	    + h0*hconstCHI1[1] + h1*hconstCHI1[2] + h2*hconstCHI1[3] 
	    + h3*hconstCHI1[4] + h4*hconstCHI1[5] + h5*hconstCHI1[6];
	chi2 += c*c;
	c = hconstCHI2[0] 
	    + h0*hconstCHI2[1] + h1*hconstCHI2[2] + h2*hconstCHI2[3] 
	    + h3*hconstCHI2[4] + h4*hconstCHI2[5] + h5*hconstCHI2[6];
	*(data_out+4*idx+3) = chi2 + c*c;
    }
}

void CPU_Trigger(unsigned int *data_in, unsigned int *data_out, int nwords)
{

    int idx;
    for (idx=0; idx<nwords; idx++) {

	unsigned int c, chi2;
	unsigned int h0 = (*(data_in+idx) & 0x0000ffff);
	unsigned int h1 = (*(data_in+idx) & 0x0000ffff) >> (4*1);
	unsigned int h2 = (*(data_in+idx) & 0x0000ffff) >> (4*2);
	unsigned int h3 = (*(data_in+idx) & 0x0000ffff) >> (4*3);
	unsigned int h4 = (*(data_in+idx) & 0x0000ffff) >> (4*4);
	unsigned int h5 = (*(data_in+idx) & 0x0000ffff) >> (4*5);

	*(data_out+4*idx)   = hconstD0[0]  
	    + h0*hconstD0[1]  + h1*hconstD0[2]  + h2*hconstD0[3]  
	    + h3*hconstD0[4]  + h4*hconstD0[5]  + h5*hconstD0[6];
	*(data_out+4*idx+1) = hconstPT[0]  
	    + h0*hconstPT[1]  + h1*hconstPT[2]  + h2*hconstPT[3]  
	    + h3*hconstPT[4]  + h4*hconstPT[5]  + h5*hconstPT[6];
	*(data_out+4*idx+2) = hconstPHI[0] 
	    + h0*hconstPHI[1] + h1*hconstPHI[2] + h2*hconstPHI[3] 
	    + h3*hconstPHI[4] + h4*hconstPHI[5] + h5*hconstPHI[6];
	c = hconstCHI0[0] 
	    + h0*hconstCHI0[1] + h1*hconstCHI0[2] + h2*hconstCHI0[3] 
	    + h3*hconstCHI0[4] + h4*hconstCHI0[5] + h5*hconstCHI0[6];
	chi2  = c*c;
	c = hconstCHI1[0] 
	    + h0*hconstCHI1[1] + h1*hconstCHI1[2] + h2*hconstCHI1[3] 
	    + h3*hconstCHI1[4] + h4*hconstCHI1[5] + h5*hconstCHI1[6];
	chi2 += c*c;
	c = hconstCHI2[0] 
	    + h0*hconstCHI2[1] + h1*hconstCHI2[2] + h2*hconstCHI2[3] 
	    + h3*hconstCHI2[4] + h4*hconstCHI2[5] + h5*hconstCHI2[6];
	*(data_out+4*idx+3) = chi2 + c*c;
    }
}

void CPU_Trigger_2(unsigned int *data_in, unsigned int *data_out, int nwords)
{

    int idx;
    for (idx=0; idx<nwords; idx++) {

	unsigned int c, chi2;
	unsigned int h0 = (*(data_in+idx) & 0x000ffff0) >> (4*0+1);
	unsigned int h1 = (*(data_in+idx) & 0x000ffff0) >> (4*1+1);
	unsigned int h2 = (*(data_in+idx) & 0x000ffff0) >> (4*2+1);
	unsigned int h3 = (*(data_in+idx) & 0x000ffff0) >> (4*3+1);
	unsigned int h4 = (*(data_in+idx) & 0x000ffff0) >> (4*4+1);
	unsigned int h5 = (*(data_in+idx) & 0x000ffff0) >> (4*5+1);

	*(data_out+4*idx)   = hconstD0[0]  
	    + h0*hconstD0[1]  + h1*hconstD0[2]  + h2*hconstD0[3]  
	    + h3*hconstD0[4]  + h4*hconstD0[5]  + h5*hconstD0[6];
	*(data_out+4*idx+1) = hconstPT[0]  
	    + h0*hconstPT[1]  + h1*hconstPT[2]  + h2*hconstPT[3]  
	    + h3*hconstPT[4]  + h4*hconstPT[5]  + h5*hconstPT[6];
	*(data_out+4*idx+2) = hconstPHI[0] 
	    + h0*hconstPHI[1] + h1*hconstPHI[2] + h2*hconstPHI[3] 
	    + h3*hconstPHI[4] + h4*hconstPHI[5] + h5*hconstPHI[6];
	c = hconstCHI0[0] 
	    + h0*hconstCHI0[1] + h1*hconstCHI0[2] + h2*hconstCHI0[3] 
	    + h3*hconstCHI0[4] + h4*hconstCHI0[5] + h5*hconstCHI0[6];
	chi2  = c*c;
	c = hconstCHI1[0] 
	    + h0*hconstCHI1[1] + h1*hconstCHI1[2] + h2*hconstCHI1[3] 
	    + h3*hconstCHI1[4] + h4*hconstCHI1[5] + h5*hconstCHI1[6];
	chi2 += c*c;
	c = hconstCHI2[0] 
	    + h0*hconstCHI2[1] + h1*hconstCHI2[2] + h2*hconstCHI2[3] 
	    + h3*hconstCHI2[4] + h4*hconstCHI2[5] + h5*hconstCHI2[6];
	*(data_out+4*idx+3) = chi2 + c*c;
    }
}

void CPU_Trigger_3(unsigned int *data_in, unsigned int *data_out, int nwords)
{

    int idx;
    for (idx=0; idx<nwords; idx++) {

	unsigned int c, chi2;
	unsigned int h0 = (*(data_in+idx) & 0x00ffff00) >> (4*0+2);
	unsigned int h1 = (*(data_in+idx) & 0x00ffff00) >> (4*1+2);
	unsigned int h2 = (*(data_in+idx) & 0x00ffff00) >> (4*2+2);
	unsigned int h3 = (*(data_in+idx) & 0x00ffff00) >> (4*3+2);
	unsigned int h4 = (*(data_in+idx) & 0x00ffff00) >> (4*4+2);
	unsigned int h5 = (*(data_in+idx) & 0x00ffff00) >> (4*5+2);

	*(data_out+4*idx)   = hconstD0[0]  
	    + h0*hconstD0[1]  + h1*hconstD0[2]  + h2*hconstD0[3]  
	    + h3*hconstD0[4]  + h4*hconstD0[5]  + h5*hconstD0[6];
	*(data_out+4*idx+1) = hconstPT[0]  
	    + h0*hconstPT[1]  + h1*hconstPT[2]  + h2*hconstPT[3]  
	    + h3*hconstPT[4]  + h4*hconstPT[5]  + h5*hconstPT[6];
	*(data_out+4*idx+2) = hconstPHI[0] 
	    + h0*hconstPHI[1] + h1*hconstPHI[2] + h2*hconstPHI[3] 
	    + h3*hconstPHI[4] + h4*hconstPHI[5] + h5*hconstPHI[6];
	c = hconstCHI0[0] 
	    + h0*hconstCHI0[1] + h1*hconstCHI0[2] + h2*hconstCHI0[3] 
	    + h3*hconstCHI0[4] + h4*hconstCHI0[5] + h5*hconstCHI0[6];
	chi2  = c*c;
	c = hconstCHI1[0] 
	    + h0*hconstCHI1[1] + h1*hconstCHI1[2] + h2*hconstCHI1[3] 
	    + h3*hconstCHI1[4] + h4*hconstCHI1[5] + h5*hconstCHI1[6];
	chi2 += c*c;
	c = hconstCHI2[0] 
	    + h0*hconstCHI2[1] + h1*hconstCHI2[2] + h2*hconstCHI2[3] 
	    + h3*hconstCHI2[4] + h4*hconstCHI2[5] + h5*hconstCHI2[6];
	*(data_out+4*idx+3) = chi2 + c*c;
    }
}

void CPU_Trigger_4(unsigned int *data_in, unsigned int *data_out, int nwords)
{

    int idx;
    for (idx=0; idx<nwords; idx++) {

	unsigned int c, chi2;
	unsigned int h0 = (*(data_in+idx) & 0x0ffff000) >> (4*0+3);
	unsigned int h1 = (*(data_in+idx) & 0x0ffff000) >> (4*1+3);
	unsigned int h2 = (*(data_in+idx) & 0x0ffff000) >> (4*2+3);
	unsigned int h3 = (*(data_in+idx) & 0x0ffff000) >> (4*3+3);
	unsigned int h4 = (*(data_in+idx) & 0x0ffff000) >> (4*4+3);
	unsigned int h5 = (*(data_in+idx) & 0x0ffff000) >> (4*5+3);

	*(data_out+4*idx)   = hconstD0[0]  
	    + h0*hconstD0[1]  + h1*hconstD0[2]  + h2*hconstD0[3]  
	    + h3*hconstD0[4]  + h4*hconstD0[5]  + h5*hconstD0[6];
	*(data_out+4*idx+1) = hconstPT[0]  
	    + h0*hconstPT[1]  + h1*hconstPT[2]  + h2*hconstPT[3]  
	    + h3*hconstPT[4]  + h4*hconstPT[5]  + h5*hconstPT[6];
	*(data_out+4*idx+2) = hconstPHI[0] 
	    + h0*hconstPHI[1] + h1*hconstPHI[2] + h2*hconstPHI[3] 
	    + h3*hconstPHI[4] + h4*hconstPHI[5] + h5*hconstPHI[6];
	c = hconstCHI0[0] 
	    + h0*hconstCHI0[1] + h1*hconstCHI0[2] + h2*hconstCHI0[3] 
	    + h3*hconstCHI0[4] + h4*hconstCHI0[5] + h5*hconstCHI0[6];
	chi2  = c*c;
	c = hconstCHI1[0] 
	    + h0*hconstCHI1[1] + h1*hconstCHI1[2] + h2*hconstCHI1[3] 
	    + h3*hconstCHI1[4] + h4*hconstCHI1[5] + h5*hconstCHI1[6];
	chi2 += c*c;
	c = hconstCHI2[0] 
	    + h0*hconstCHI2[1] + h1*hconstCHI2[2] + h2*hconstCHI2[3] 
	    + h3*hconstCHI2[4] + h4*hconstCHI2[5] + h5*hconstCHI2[6];
	*(data_out+4*idx+3) = chi2 + c*c;
    }
}

void CPU_Trigger_5(unsigned int *data_in, unsigned int *data_out, int nwords)
{

    int idx;
    for (idx=0; idx<nwords; idx++) {

	unsigned int c, chi2;
	unsigned int h0 = (*(data_in+idx) & 0xffff0000) >> (4*0+4);
	unsigned int h1 = (*(data_in+idx) & 0xffff0000) >> (4*1+4);
	unsigned int h2 = (*(data_in+idx) & 0xffff0000) >> (4*2+4);
	unsigned int h3 = (*(data_in+idx) & 0xffff0000) >> (4*3+4);
	unsigned int h4 = (*(data_in+idx) & 0xffff0000) >> (4*4+4);
	unsigned int h5 = (*(data_in+idx) & 0xffff0000) >> (4*5+4);

	*(data_out+4*idx)   = hconstD0[0]  
	    + h0*hconstD0[1]  + h1*hconstD0[2]  + h2*hconstD0[3]  
	    + h3*hconstD0[4]  + h4*hconstD0[5]  + h5*hconstD0[6];
	*(data_out+4*idx+1) = hconstPT[0]  
	    + h0*hconstPT[1]  + h1*hconstPT[2]  + h2*hconstPT[3]  
	    + h3*hconstPT[4]  + h4*hconstPT[5]  + h5*hconstPT[6];
	*(data_out+4*idx+2) = hconstPHI[0] 
	    + h0*hconstPHI[1] + h1*hconstPHI[2] + h2*hconstPHI[3] 
	    + h3*hconstPHI[4] + h4*hconstPHI[5] + h5*hconstPHI[6];
	c = hconstCHI0[0] 
	    + h0*hconstCHI0[1] + h1*hconstCHI0[2] + h2*hconstCHI0[3] 
	    + h3*hconstCHI0[4] + h4*hconstCHI0[5] + h5*hconstCHI0[6];
	chi2  = c*c;
	c = hconstCHI1[0] 
	    + h0*hconstCHI1[1] + h1*hconstCHI1[2] + h2*hconstCHI1[3] 
	    + h3*hconstCHI1[4] + h4*hconstCHI1[5] + h5*hconstCHI1[6];
	chi2 += c*c;
	c = hconstCHI2[0] 
	    + h0*hconstCHI2[1] + h1*hconstCHI2[2] + h2*hconstCHI2[3] 
	    + h3*hconstCHI2[4] + h4*hconstCHI2[5] + h5*hconstCHI2[6];
	*(data_out+4*idx+3) = chi2 + c*c;
    }
}
