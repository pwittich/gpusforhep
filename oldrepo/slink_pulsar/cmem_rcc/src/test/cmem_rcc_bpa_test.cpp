/****************************************************************/
/*								*/
/*  This is a test program for the CMEM_RCC driver & library	*/
/*								*/
/*  12. Dec. 01  MAJO  created					*/
/*								*/
/***********C 2001 - A nickel program worth a dime***************/


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "rcc_error/rcc_error.h"
#include "cmem_rcc/cmem_rcc.h"
#include "ROSGetInput/get_input.h"

int main(void)
{
  unsigned int paddr, uaddr, asize, size1 = 0x1000, size2 = 0x10000, ret, loop, *ptr;
  int handle2, handle;
  char dummy[20], name1[CMEM_MAX_NAME], name2[CMEM_MAX_NAME];
  
  ret = CMEM_Open();
  if (ret)
  {
    rcc_error_print(stdout, ret);
    exit(-1);
  }
  
  printf("Testing BigPhysArea based memory allocation\n");
  printf("Enter the size [in bytes] of the first buffer: ");
  size1 = gethexd(size1);
  printf("Enter the name of the first buffer: ");
  getstrd(name1, "cmem_rcc_test_1");

  printf("Enter the size [in bytes] of the second buffer: ");
  size2 = gethexd(size2);
  printf("Enter the name of the second buffer: ");
  getstrd(name2, "cmem_rcc_test_2");

  ret = CMEM_BPASegmentAllocate(size1, name1, &handle);
  if (ret)
  {
    rcc_error_print(stdout, ret);
    CMEM_Close();
    exit(-1);
  }
  printf("First CMEM_BPASegmentAllocate returns handle = %d\n",handle);

  ret = CMEM_BPASegmentAllocate(size2, name2, &handle2);
  if (ret)
  {
    rcc_error_print(stdout, ret);
    CMEM_Close();
    exit(-1);
  }
  printf("Second CMEM_BPASegmentAllocate returns handle = %d\n",handle2);
  
  ret = CMEM_SegmentVirtualAddress(handle, &uaddr);
  if (ret)
    rcc_error_print(stdout, ret);
  
  ret = CMEM_SegmentPhysicalAddress(handle, &paddr);
  if (ret)
    rcc_error_print(stdout, ret);
  
  ret = CMEM_SegmentSize(handle, &asize);
  if (ret)
    rcc_error_print(stdout, ret);
 
  printf("First segment:\n");
  printf("Physical address = 0x%08x\n", paddr);
  printf("Virtual address  = 0x%08x\n", uaddr);
  printf("Actual size      = 0x%08x\n", asize);
  
  ret = CMEM_SegmentVirtualAddress(handle2, &uaddr);
  if (ret)
    rcc_error_print(stdout, ret);
  
  ret = CMEM_SegmentPhysicalAddress(handle2, &paddr);
  if (ret)
    rcc_error_print(stdout, ret);
  
  ret = CMEM_SegmentSize(handle2, &asize);
  if (ret)
    rcc_error_print(stdout, ret);
 
  printf("Second segment:\n");
  printf("Physical address = 0x%08x\n", paddr);
  printf("Virtual address  = 0x%08x\n", uaddr);
  printf("Actual size      = 0x%08x\n", asize);
  
  //test the entire buffer
  ptr = (unsigned int *)uaddr;
  for(loop = 0; loop < (size2 >> 2); loop++)
  {
    *ptr = loop;
    if (loop != *ptr)
      printf("ERROR: Failed to write to virt. addr 0x%08x\n", (unsigned int)ptr); 
    ptr++;
  }

  CMEM_Dump();
  printf("Press <return> to release the buffer\n");
  fgets(dummy, 10, stdin);
     
  ret = CMEM_BPASegmentFree(handle);
  if (ret)
    rcc_error_print(stdout, ret);
  printf("handle %d returned\n",handle);
     
  ret = CMEM_BPASegmentFree(handle2);
  if (ret)
    rcc_error_print(stdout, ret);
  printf("handle %d returned\n",handle2);
  
  ret = CMEM_Close();
  if (ret)
  {
    rcc_error_print(stdout, ret);
    exit(-1);
  }
  return(0);
}
