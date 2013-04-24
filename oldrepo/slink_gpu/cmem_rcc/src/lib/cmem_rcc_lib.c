/************************************************************************/
/*									*/
/*  This is the CMEM_RCC library 					*/
/*  Its purpose is to provide user applications with contiguous data 	*/
/*  buffers for DMA operations. It is not based on any extensions to 	*/
/*  the Linux kernel like the BigPhysArea patch.			*/
/*									*/
/*  12. Dec. 01  MAJO  created						*/
/*									*/
/*******C 2001 - The software with that certain something****************/

#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#ifdef L2GPU
#include <asm-generic/page.h>
#include <asm/mman.h>
#else
#include <asm/page.h>
#endif 
#include "rcc_error/rcc_error.h"
#include "cmem_rcc/cmem_rcc.h"
#include "cmem_rcc/cmem_rcc_drv.h"

//Globals
static int fd, is_open=0;
cmem_rcc_t desc[MAX_CMEM_BUFFERS];

/**************************/
unsigned int CMEM_Open(void)
/**************************/
{
  int ret, loop;
  
  //we need to open the driver only once
  if (is_open)
  {
    is_open++;             //keep track of multiple open calls
    return(RCC_ERROR_RETURN(0, CMEM_RCC_SUCCESS));
  }

  //open the error package
  ret = rcc_error_init(P_ID_CMEM_RCC, CMEM_err_get);
  if (ret)
  {
    debug(("CMEM_Open: Failed to open error package\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_ERROR_FAIL)); 
  }
  debug(("CMEM_Open: error package opened\n")); 

  debug(("CMEM_Open: Opening /dev/cmem_rcc\n"));
  if ((fd = open("/dev/cmem_rcc", O_RDWR)) < 0)
  {
    debug(("CMEM_Open: Failed to open /dev/cmem_rcc\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_FILE)); 
  }
  debug(("CMEM_Open: /dev/cmem_rcc is open\n"));
  
  //Initialize the array of buffers
  for (loop = 0; loop < MAX_CMEM_BUFFERS; loop++)
    desc[loop].used = 0;
  
  is_open = 1;
  
return(0);
}


/***************************/
unsigned int CMEM_Close(void)
/***************************/
{  
  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
  
  if (is_open > 1)
    is_open--;
  else
  {
    close(fd);
    is_open = 0;
  }
  
  return(0);
}


/***************************************************************************************/
unsigned int CMEM_SegmentAllocate(unsigned int size, char *name, int *segment_identifier)
/***************************************************************************************/
{ 
  int order, ret, loop, ok;
  
  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
 
  if (size == 0)
  {
    debug(("CMEM_SegmentAllocate: size is zero\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_NOSIZE));
  }
 
  //Look for a free descriptor
  ok = 0;
  for (loop = 0; loop < MAX_CMEM_BUFFERS; loop++)
  {
    if(desc[loop].used == 0)
    {
      ok = 1;
      break;
    }
  }
  
  if (!ok)
  {
    debug(("CMEM_SegmentAllocate: No free descriptor found\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_OVERFLOW));
  }
  
  debug(("CMEM_SegmentAllocate: Using descriptor %d\n", loop));
  
  //Calculate the "order"
  if (size <= PAGE_SIZE)            order = 0;
  else if (size <= 2 * PAGE_SIZE)   order = 1;
  else if (size <= 4 * PAGE_SIZE)   order = 2;
  else if (size <= 8 * PAGE_SIZE)   order = 3;
  else if (size <= 16 * PAGE_SIZE)  order = 4;
  else if (size <= 32 * PAGE_SIZE)  order = 5;
  else if (size <= 64 * PAGE_SIZE)  order = 6;
  else if (size <= 128 * PAGE_SIZE) order = 7;
  else if (size <= 256 * PAGE_SIZE) order = 8;
  else if (size <= 512 * PAGE_SIZE) order = 9;
  else 
  {
    debug(("CMEM_SegmentAllocate: size is too big\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_TOOBIG));
  }

  strcpy(desc[loop].name, name);
  desc[loop].order = order;

  debug(("CMEM_SegmentAllocate: order = %d\n", order));
  
  debug(("CMEM_SegmentAllocate: calling ioctl\n"));    
  ret = ioctl(fd, CMEM_RCC_GET, &desc[loop]);
  if (ret)
  {
    debug(("CMEM_SegmentAllocate: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_IOCTL)); 
  }

  debug(("CMEM_SegmentAllocate: calling mmap\n"));    
  debug(("CMEM_SegmentAllocate: size   = 0x%08x\n",desc[loop].size));
  debug(("CMEM_SegmentAllocate: offset = 0x%08x\n",desc[loop].paddr));
  desc[loop].uaddr = (unsigned int)mmap(0, desc[loop].size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_32BIT, fd, (int)desc[loop].paddr);
  if ((int)desc[loop].uaddr == 0 || (int)desc[loop].uaddr == -1)
  {
    debug(("CMEM_SegmentAllocate: Error from mmap, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_MMAP)); 
  }
  debug(("CMEM_SegmentAllocate: Virtual address = 0x%08x\n", (unsigned int)desc[loop].uaddr)); 
  desc[loop].used = 1;
  *segment_identifier = loop;
  return(0);
}


/***************************************************/
unsigned int CMEM_SegmentFree(int segment_identifier)
/***************************************************/
{
  int ret;  
  
  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
    
  if (desc[segment_identifier].used != 1)
  {
    debug(("CMEM_SegmentFree: Invalid handle\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_ILLHAND));
  }
  
  debug(("CMEM_SegmentFree: calling munmap\n"));    
  ret = munmap((char *)desc[segment_identifier].uaddr, desc[segment_identifier].size);
  if (ret)
  {
    debug(("CMEM_SegmentFree: Error from munmap, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_MUNMAP)); 
  }
  
  debug(("CMEM_SegmentFree: calling ioctl\n"));    
  ret = ioctl(fd, CMEM_RCC_FREE, &desc[segment_identifier]);
  if (ret)
  {
    debug(("CMEM_SegmentFree: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_IOCTL)); 
  }

  desc[segment_identifier].used = 0;
  return(0);
}


/***********************************************************************************/
CMEM_Error_code_t CMEM_SegmentSize(int segment_identifier, unsigned int *actual_size)
/***********************************************************************************/
{  
  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
    
  if (desc[segment_identifier].used != 1)
  {
    debug(("CMEM_SegmentFree: Invalid handle\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_ILLHAND));
  }
  
  *actual_size = desc[segment_identifier].size;
  return(0);
}


/***************************************************************************************************/
CMEM_Error_code_t CMEM_SegmentPhysicalAddress(int segment_identifier, unsigned int *physical_address)
/***************************************************************************************************/
{  

  //  fprintf( stderr, "CMEM_SegmentPhysicalAddress: segment_identifier = %d\n", segment_identifier );

  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
    
  if (desc[segment_identifier].used != 1)
  {
    debug(("CMEM_SegmentFree: Invalid handle\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_ILLHAND));
  }

  //  fprintf( stderr, "CMEM_SegmentPhysicalAddress: before phys address\n" );
  *physical_address = desc[segment_identifier].paddr;
  //  fprintf( stderr, "CMEM_SegmentPhysicalAddress: after phys address\n" );
  return(0);
}


/*************************************************************************************************/
CMEM_Error_code_t CMEM_SegmentVirtualAddress(int segment_identifier, unsigned int *virtual_address)
/*************************************************************************************************/
{  
  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
    
  if (desc[segment_identifier].used != 1)
  {
    debug(("CMEM_SegmentFree: Invalid handle\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_ILLHAND));
  }

  *virtual_address = desc[segment_identifier].uaddr;
  return(0);
}


/*******************************/
CMEM_Error_code_t CMEM_Dump(void)
/*******************************/
{      
  int ret;
  char mytext[TEXT_SIZE];

  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);  

  ret = ioctl(fd, DUMP, mytext);
  if (ret)
  {
    debug(("CMEM_Dump: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  printf("%s", mytext);

  return(0);
}

#ifdef BPA
/******************************************************************************************/
unsigned int CMEM_BPASegmentAllocate(unsigned int size, char *name, int *segment_identifier)
/******************************************************************************************/
{ 
  int ret, loop, ok;
  
  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
  
  //Look for a free descriptor
  ok = 0;
  for (loop = 0; loop < MAX_CMEM_BUFFERS; loop++)
  {
    if(desc[loop].used == 0)
    {
      ok = 1;
      break;
    }
  }
  
  if (!ok)
  {
    debug(("CMEM_SegmentAllocate: No free descriptor found\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_OVERFLOW));
  }
  
  debug(("CMEM_BPASegmentAllocate: Using descriptor %d\n", loop));

  strcpy(desc[loop].name, name);
  desc[loop].size = size; 
  desc[loop].order = 0; //not required
  debug(("CMEM_BPASegmentAllocate: calling ioctl with size = 0x%08x\n", size));
  
  debug(("CMEM_BPASegmentAllocate: calling ioctl\n"));    
  ret = ioctl(fd, CMEM_RCC_BPA_GET, &desc[loop]);
  if (ret)
  {
    debug(("CMEM_BPASegmentAllocate: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_IOCTL)); 
  }
  
  if (!desc[loop].paddr)
  {
     debug(("CMEM_BPASegmentAllocate: Error from ioctl(PCI address = 0)\n"));
     return(RCC_ERROR_RETURN(0, CMEM_RCC_IOCTL));
  }

  debug(("CMEM_BPASegmentAllocate: calling mmap\n"));    
  debug(("CMEM_BPASegmentAllocate: size   = 0x%08x\n",desc[loop].size));
  debug(("CMEM_BPASegmentAllocate: offset = 0x%08x\n",desc[loop].paddr));
  desc[loop].uaddr = (unsigned int)mmap(0, desc[loop].size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_32BIT, fd, (int)desc[loop].paddr);
  if ((int)desc[loop].uaddr == 0 || (int)desc[loop].uaddr == -1)
  {
    debug(("CMEM_BPASegmentAllocate: Error from mmap, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_MMAP)); 
  }
  debug(("CMEM_BPASegmentAllocate: Virtual address = 0x%08x\n", (unsigned int)desc[loop].uaddr)); 
  desc[loop].used = 1;
  *segment_identifier = loop;
  return(0);
}


/******************************************************/
unsigned int CMEM_BPASegmentFree(int segment_identifier)
/******************************************************/
{
  int ret;  
  
  if (!is_open) 
    return(CMEM_RCC_NOTOPEN);
    
  if (desc[segment_identifier].used != 1)
  {
    debug(("CMEM_BPASegmentFree: Invalid handle\n"));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_ILLHAND));
  }
  
  debug(("CMEM_BPASegmentFree: calling munmap\n"));    
  ret = munmap((char *)desc[segment_identifier].uaddr, desc[segment_identifier].size);
  if (ret)
  {
    debug(("CMEM_BPASegmentFree: Error from munmap, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_MUNMAP)); 
  }
  
  debug(("CMEM_BPASegmentFree: calling ioctl\n"));    
  ret = ioctl(fd, CMEM_RCC_BPA_FREE, &desc[segment_identifier]);
  if (ret)
  {
    debug(("CMEM_BPASegmentFree: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, CMEM_RCC_IOCTL)); 
  }

  desc[segment_identifier].used = 0;
  return(0);
}

#else

/******************************************************************************************/
unsigned int CMEM_BPASegmentAllocate(unsigned int size, char *name, int *segment_identifier)
/******************************************************************************************/
{
  printf("CMEM_BPASegmentAllocate is not supported\n");
  return(RCC_ERROR_RETURN(0, CMEM_RCC_OVERFLOW));
}

/******************************************************/
unsigned int CMEM_BPASegmentFree(int segment_identifier)
/******************************************************/
{
  printf("CMEM_BPASegmentAllocate is not supported\n");
  return(RCC_ERROR_RETURN(0, CMEM_RCC_OVERFLOW));
}
#endif


/****************************************************************/
unsigned int CMEM_err_get(err_pack err, err_str pid, err_str code)
/****************************************************************/
{ 
  strcpy(pid, P_ID_CMEM_RCC_STR);

  switch (RCC_ERROR_MINOR(err))
  {  
    case CMEM_RCC_SUCCESS:  strcpy(code, CMEM_RCC_SUCCESS_STR); break;
    case CMEM_RCC_FILE:     strcpy(code, CMEM_RCC_FILE_STR); break;
    case CMEM_RCC_NOTOPEN:  strcpy(code, CMEM_RCC_NOTOPEN_STR); break;
    case CMEM_RCC_IOCTL:    strcpy(code, CMEM_RCC_IOCTL_STR); break;
    case CMEM_RCC_MMAP:     strcpy(code, CMEM_RCC_MMAP_STR); break;
    case CMEM_RCC_MUNMAP:   strcpy(code, CMEM_RCC_MUNMAP_STR); break;
    case CMEM_RCC_OVERFLOW: strcpy(code, CMEM_RCC_OVERFLOW_STR); break;
    case CMEM_RCC_TOOBIG:   strcpy(code, CMEM_RCC_TOOBIG_STR); break;
    case CMEM_RCC_ILLHAND:  strcpy(code, CMEM_RCC_ILLHAND_STR); break;
    case CMEM_RCC_NOSIZE:   strcpy(code, CMEM_RCC_NOSIZE_STR); break;
    default:                strcpy(code, CMEM_RCC_NO_CODE_STR); return(RCC_ERROR_RETURN(0,CMEM_RCC_NO_CODE)); break;
  }
  return(RCC_ERROR_RETURN(0, CMEM_RCC_SUCCESS));
}

