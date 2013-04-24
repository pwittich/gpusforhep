/****************************************************************/
/*								*/
/*  This is a test program for the CMEM_RCC driver & library	*/
/*								*/
/*  11. Jun. 02  MAJO  created					*/
/*								*/
/***********C 2002 - A nickel program worth a dime***************/


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
  unsigned int cnt, asize, size = 0x1000, ret;
  int handle;
  char dummy[20], name[CMEM_MAX_NAME];
  
  ret = CMEM_Open();
  if (ret)
  {
    rcc_error_print(stdout, ret);
    exit(-1);
  }
  
  printf("Enter the size [in bytes] of the buffers to be allocated: ");
  size = gethexd(size);

  sprintf(name,"tseg");
  ret = CMEM_SegmentAllocate(size, name, &handle);
  if (ret)
  {
    rcc_error_print(stdout, ret);
    CMEM_Close();
    exit(-1);
  }
  
  ret = CMEM_SegmentSize(handle, &asize);
  if (ret)
    rcc_error_print(stdout, ret);

  printf("Actual size of first buffer = 0x%08x\n", asize); 
  
  cnt = 0;
  while(ret == 0)
  {
    cnt++;
    sprintf(name,"tseg%d",cnt);
    ret = CMEM_SegmentAllocate(size, name, &handle);
    if (!(cnt % 100))
      printf("cnt = %d\n", cnt);
  }
  rcc_error_print(stdout, ret);
  printf("%d buffers allocated\n",cnt);
  printf("Total buffer size = %f MB\n",(float)asize*(float)cnt/1024.0/1024.0);
  printf("Press <return> to release the buffers\n");
  gets(dummy);

  ret = CMEM_Close();
  if (ret)
  {
    rcc_error_print(stdout, ret);
    exit(-1);
  }
  return(0);
}
