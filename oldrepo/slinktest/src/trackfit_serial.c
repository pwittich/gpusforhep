const unsigned int cpu_constD0[7]   =	{0, 1, 0, 0, 0, 0, 0};
const unsigned int cpu_constPT[7]   =	{0, 0, 1, 0, 0, 0, 0};
const unsigned int cpu_constPHI[7]  =	{0, 0, 0, 1, 0, 0, 0};
const unsigned int cpu_constCHI0[7] =	{0, 0, 0, 0, 1, 0, 0};
const unsigned int cpu_constCHI1[7] =	{0, 0, 0, 0, 0, 1, 0};
const unsigned int cpu_constCHI2[7] =	{0, 0, 0, 0, 0, 0, 1};


    cudaEventRecord(start_event0, 0);  
    long idx;
    unsigned int c, chi2, h0, h1, h2, h3, h4, h5;
    for(idx=0; idx<N_RECORDS_IN; idx++)
      {
	h0 = h_HIT0[idx];
	h1 = h_HIT1[idx];
	h2 = h_HIT2[idx];
	h3 = h_HIT3[idx];
	h4 = h_HIT4[idx];
	h5 = h_HIT5[idx];
	h_D0[idx]  = cpu_constD0[0]
	  + h0*cpu_constD0[1]  + h1*cpu_constD0[2]  + h2*cpu_constD0[3]  
	  + h3*cpu_constD0[4]  + h4*cpu_constD0[5]  + h5*cpu_constD0[6];
	h_PT[idx]  = cpu_constPT[0]  
	  + h0*cpu_constPT[1]  + h1*cpu_constPT[2]  + h2*cpu_constPT[3]  
	  + h3*cpu_constPT[4]  + h4*cpu_constPT[5]  + h5*cpu_constPT[6];
	h_PHI[idx] = cpu_constPHI[0] 
	  + h0*cpu_constPHI[1] + h1*cpu_constPHI[2] + h2*cpu_constPHI[3] 
	  + h3*cpu_constPHI[4] + h4*cpu_constPHI[5] + h5*cpu_constPHI[6];
	c = cpu_constCHI0[0] 
	  + h0*cpu_constCHI0[1] + h1*cpu_constCHI0[2] + h2*cpu_constCHI0[3] 
	  + h3*cpu_constCHI0[4] + h4*cpu_constCHI0[5] + h5*cpu_constCHI0[6];
	chi2  = c*c;
	c = cpu_constCHI1[0] 
	  + h0*cpu_constCHI1[1] + h1*cpu_constCHI1[2] + h2*cpu_constCHI1[3] 
	  + h3*cpu_constCHI1[4] + h4*cpu_constCHI1[5] + h5*cpu_constCHI1[6];
	chi2 += c*c;
	c = cpu_constCHI2[0] 
	  + h0*cpu_constCHI2[1] + h1*cpu_constCHI2[2] + h2*cpu_constCHI2[3] 
	  + h3*cpu_constCHI2[4] + h4*cpu_constCHI2[5] + h5*cpu_constCHI2[6];
	h_CHI[idx] = chi2 + c*c;
      }
    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);   // block until the event is actually recorded
    cudaEventElapsedTime(&time_elapsed0, start_event0, stop_event0);
    printf("\nCPU calc: %.2f milliseconds\n\n", time_elapsed0);
    
    printf("h_D0[0]=%u\n", h_D0[0]);
    printf("h_CHI[0]=%u\n", h_CHI[0]);
    
    n++;
    if(n==1) break;
