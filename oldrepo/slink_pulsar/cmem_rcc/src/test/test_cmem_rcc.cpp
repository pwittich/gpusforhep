/****************************************************************/
/*								*/
/*  This is the test program for the CMEM_RCC driver & library	*/
/*								*/
/*  12. Dec. 01  MAJO  created					*/
/*								*/
/***********C 2001 - A nickel program worth a dime***************/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "rcc_error/rcc_error.h"
#include "cmem_rcc/cmem_rcc.h"
#include "ROSGetInput/get_input.h"

int cont;


// Prototypes
int stresstest(int mode);
void SigQuitHandler(int signum);


/*****************************/
void SigQuitHandler(int signum)
/*****************************/
{
  cont=0;
}


/************/
int main(void)
/************/
{
  unsigned int paddr, uaddr, asize, size = 0x1000, ret;
  int stat, fun, handle;
  char name[CMEM_MAX_NAME];
  static struct sigaction sa;

  sigemptyset(&sa.sa_mask); 
  sa.sa_flags = 0; 
  sa.sa_handler = SigQuitHandler;
  stat = sigaction(SIGQUIT, &sa, NULL);
  if (stat < 0)
  {
    printf("Cannot install signal handler (error=%d)\n", stat);
    exit(0);
  }
  
  fun = 1;
  
  while (fun != 0)  
  {
    printf("\n");
    printf("Select an option:\n");
    printf("   1 CMEM_Open                 2 CMEM_Close\n");  
    printf("   3 CMEM_SegmentAllocate      4 CMEM_SegmentFree\n");
    printf("   5 CMEM_BPASegmentAllocate   6 CMEM_BPASegmentFree\n");
    printf("   7 CMEM_SegmentSize          8 CMEM_SegmentPhysicalAddress\n");
    printf("   9 CMEM_Dump                10 CMEM_SegmentVirtualAddress\n");
    printf("============================================================\n");
    printf("  11 Stress test __get_free_pages\n");
    printf("  12 Stress test BPA\n");
    printf("  13 Stress test __get_free_pages and BPA\n");
    printf("   0 Exit\n");
    printf("Your choice ");
    fun = getdecd(fun);
    if (fun == 1)
    {  
      ret = CMEM_Open();
      if (ret)
        rcc_error_print(stdout, ret);
    }
     
    if (fun == 2) 
    {
      ret = CMEM_Close();
      if (ret)
        rcc_error_print(stdout, ret);
    }    
    
    if (fun == 3) 
    {
      printf("Enter the size [in bytes] of the segment to be allocated: ");
      size = gethexd(size);

      printf("Enter the name of the segment: ");
      getstrd(name, "cmem_rcc_test");

      ret = CMEM_SegmentAllocate(size, name, &handle);
      if (ret)
        rcc_error_print(stdout, ret);
      printf("The handle is %d\n", handle);
    }    
    
    if (fun == 4) 
    {
      printf("Enter the handle: ");
      handle = getdecd(handle);
      
      ret = CMEM_SegmentFree(handle);
      if (ret)
        rcc_error_print(stdout, ret);
      printf("handle %d returned\n", handle);
    }    
    
    if (fun == 5) 
    {
      printf("Enter the size [in bytes] of the segment to be allocated: ");
      size = gethexd(size);

      printf("Enter the name of the segment: ");
      getstrd(name, "cmem_rcc_bpa_test");

      ret = CMEM_BPASegmentAllocate(size, name, &handle);
      if (ret)
        rcc_error_print(stdout, ret);
      printf("The handle is %d\n", handle);
    }
    
    if (fun == 6) 
    {
      printf("Enter the handle: ");
      handle = getdecd(handle);
      
      ret = CMEM_BPASegmentFree(handle);
      if (ret)
        rcc_error_print(stdout, ret);
      printf("handle %d returned\n", handle);
    }
    
    if (fun == 7) 
    { 
      printf("Enter the handle: ");
      handle = getdecd(handle);
      ret = CMEM_SegmentSize(handle, &asize);
      if (ret)
        rcc_error_print(stdout, ret);
      printf("Segment size = 0x%08x bytes\n", asize);
    }
    
    if (fun == 8) 
    {
      printf("Enter the handle: ");
      handle = getdecd(handle);  
      ret = CMEM_SegmentVirtualAddress(handle, &uaddr);
      if (ret)
        rcc_error_print(stdout, ret);
      printf("Segment virtual address = 0x%08x bytes\n", uaddr);
    }

    if (fun == 9) 
    {
      CMEM_Dump();
    }

    if (fun == 10) 
    {
      printf("Enter the handle: ");
      handle = getdecd(handle);
      ret = CMEM_SegmentPhysicalAddress(handle, &paddr);
      if (ret)
        rcc_error_print(stdout, ret); 
      printf("Segment physical address = 0x%08x bytes\n", paddr);
    }
    if (fun == 11)
      stresstest(1); 
    if (fun == 12) 
      stresstest(2);
    if (fun == 13) 
      stresstest(3);
    
  }
  return(0);
}


/**********************/
int stresstest(int mode)
/**********************/
{
  unsigned int n1, buffs, num, dummy, size, ret, loop;
  int handle1[100],handle2[100];
       
  ret = CMEM_Open();
  if (ret)
  {
    rcc_error_print(stdout, ret);
    return(-1);
  }      
  
  printf("Enther number of loops (0=run forever) \n");
  num = getdecd(1);

  printf("Enther number of concurrent buffers (max. 100) \n");
  buffs  = getdecd(10);
  
  cont = 1;
  n1 = 0;
  printf("Test running. Press <ctrl+\\> to stop\n");
  while(cont)
  { 
    for(loop = 0; loop < buffs; loop++)
    {
      size = rand() % 20000;
      if (size == 0)
        size = 100;

      if (mode == 1 || mode == 3)
      {
        ret = CMEM_SegmentAllocate(size, "cmem_rcc_stress", &handle1[loop]);
        if (ret) { rcc_error_print(stdout, ret); return(-1); } 
        ret = CMEM_SegmentSize(handle1[loop], &dummy);
        if (ret) { rcc_error_print(stdout, ret); return(-1); }       
        ret = CMEM_SegmentVirtualAddress(handle1[loop], &dummy);
        if (ret) { rcc_error_print(stdout, ret); return(-1); }      
        ret = CMEM_SegmentPhysicalAddress(handle1[loop], &dummy);     
        if (ret) { rcc_error_print(stdout, ret); return(-1); }    
      }
      
      if (mode == 2 || mode == 3)
      {
        ret = CMEM_BPASegmentAllocate(size, "cmem_rcc_stress", &handle2[loop]);
        if (ret) { rcc_error_print(stdout, ret); return(-1); } 
        ret = CMEM_SegmentSize(handle2[loop], &dummy);
        if (ret) { rcc_error_print(stdout, ret); return(-1); }       
        ret = CMEM_SegmentVirtualAddress(handle2[loop], &dummy);
        if (ret) { rcc_error_print(stdout, ret); return(-1); }      
        ret = CMEM_SegmentPhysicalAddress(handle2[loop], &dummy);     
        if (ret) { rcc_error_print(stdout, ret); return(-1); }    
      }
    }
 
    for(loop = 0; loop < buffs; loop++)
    {
      if (mode == 1 || mode == 3)
        ret = CMEM_SegmentFree(handle1[loop]);
      if (mode == 2 || mode == 3)
        ret = CMEM_BPASegmentFree(handle2[loop]);
      if (ret) { rcc_error_print(stdout, ret); return(-1); } 
    }
    
  if (num > 1)
    num--;
  if (num == 1)
    break;  
  n1++;
  }
 
  printf("%d loops execuded\n", n1); 
  CMEM_Dump();

  ret = CMEM_Close();
  if (ret)
  {
    rcc_error_print(stdout, ret);
    return(-1);
  }

  return(0);
}






