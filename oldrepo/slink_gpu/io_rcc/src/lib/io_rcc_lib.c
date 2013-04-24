/************************************************************************/
/*									*/
/*  This is the IO_RCC library 	   				        */
/*  Its purpose is to provide user applications with access to  	*/
/*  PCI MEM an PC I/O space 						*/
/*									*/
/*   6. Jun. 02  MAJO  created						*/
/*									*/
/*******C 2002 - The software with that certain something****************/

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
#include <asm-generic/page.h>
#include <asm/mman.h>
#include "rcc_error/rcc_error.h"
#include "io_rcc/io_rcc.h"
#include "io_rcc/io_rcc_driver.h"

//Globals
static int fd, is_open=0;


/**************************/
IO_ErrorCode_t IO_Open(void)
/**************************/
{  
  int ret;
  
  //we need to open the driver only once
  if (is_open)
  {
    is_open++;             //keep track of multiple open calls
    return(RCC_ERROR_RETURN(0, IO_RCC_SUCCESS));
  }

  //open the error package
  ret = rcc_error_init(P_ID_IO_RCC, IO_RCC_err_get);
  if (ret)
  {
    debug(("IO_Open: Failed to open error package\n"));
    return(RCC_ERROR_RETURN(0, IO_RCC_ERROR_FAIL)); 
  }
  debug(("IO_Open: error package opened\n")); 

  debug(("IO_Open: Opening /dev/io_rcc\n"));
  if ((fd = open("/dev/io_rcc", O_RDWR)) < 0)
  {
    perror("open");
    return(RCC_ERROR_RETURN(0, IO_RCC_FILE)); 
  }
  debug(("IO_Open: /dev/io_rcc is open\n"));
  
  is_open = 1;
  
  return(IO_RCC_SUCCESS);
}


/***************************/
IO_ErrorCode_t IO_Close(void)
/***************************/
{
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  if (is_open > 1)
    is_open--;
  else
  {
    close(fd);
    is_open = 0;
  }
  
  return(IO_RCC_SUCCESS);
}


/***********************************************************************/
IO_ErrorCode_t IO_PCIMemMap(u_int pci_addr, u_int size, u_int *virt_addr)
/***********************************************************************/
{
  int vaddr, offset;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));

  fprintf(stderr,"size: %d\tpciaddr: 0x%x\n",size,pci_addr);
  //offset = 0x0;
  //  offset = pci_addr;
  offset = pci_addr & 0xfff;   //mmap seem to need 4K alignment
  //offset = pci_addr & 0x7ff;   //wild guess
  fprintf(stderr, "~(sysconf(_SC_PAGE_SIZE)-1): 0x%8x\n", 
	  ~(sysconf(_SC_PAGE_SIZE)-1)  );
  pci_addr &= ~(sysconf(_SC_PAGE_SIZE)-1); // see ex in mmap man

  fprintf(stderr,"IO_PCIMemMap: before mmap\n");
  fprintf(stderr,"size: %d\tpci_addr: 0x%x\toffset: 0x%x\n", size, pci_addr,offset);

  if(fd==-1) fprintf(stderr,"Error in opening file for writing...\n");
  else{ fprintf(stderr,"Opened file OK...fd=%d\n",fd); }

  //vaddr = (int)mmap();
  // orig
  //  vaddr = (int)mmap(0, size, (PROT_READ|PROT_WRITE), MAP_SHARED|MAP_32BIT, fd, (int)pci_addr);
  vaddr = (int)mmap(0, size, (PROT_READ|PROT_WRITE), MAP_SHARED, fd, (int)pci_addr);
    fprintf(stderr,"KH:: 1st vaddr: 0x%x\n", vaddr);

  //vaddr = (int)mmap(fd,0,size,(PROT_READ|PROT_WRITE),MAP_SHARED|MAP_32BIT,(int)pci_addr);
  fprintf(stderr, "size: %d\tpci_addr: 0x%x\n", size, pci_addr );
  // orig
  //  if (vaddr <= 0)
  fprintf(stderr, "before check ...\n"  );
  if (vaddr == MAP_FAILED)
  {
    fprintf(stderr,"--> vaddr: 0x%x\n", vaddr);
    fprintf(stderr,"---> errno %d\n", errno);
    fprintf(stderr,"---> strerror %s\n", strerror(errno));
    fprintf(stderr,"---> perror\n");
    perror(NULL);
    debug(("IO_PCIMemMap: Error from mmap\n"));
    *virt_addr = 0;
    return(RCC_ERROR_RETURN(0, IO_RCC_MMAP));
  }    
  fprintf(stderr,"IO_PCIMemMap: after mmap\n");
  
  *virt_addr = vaddr + offset;
  fprintf(stderr,"IO_PCIMemMap: virt_addr = 0x%x\n", *virt_addr);
  debug(("IO_PCMemMap: virtual address = 0x%08x\n",*virt_addr));
  return(IO_RCC_SUCCESS);
}


/********************************************************/
IO_ErrorCode_t IO_PCIMemUnmap(u_int virt_addr, u_int size)
/********************************************************/
{
  int ret;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  ret = munmap((char *)(virt_addr & 0xfffff000), size);
  if (ret)
  {
    debug(("IO_PCIMemUnmap: Error from munmap, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, IO_RCC_MUNMAP));
  }

  return(IO_RCC_SUCCESS);
}


/******************************************************/
IO_ErrorCode_t IO_IOPeekUInt(u_int address, u_int *data)
/******************************************************/
{
  int ret;
  IO_RCC_IO_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  if (address & 0x3)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));
  
  params.offset = address;
  params.size = 4;
  
  ret = ioctl(fd, IOPEEK, &params);
  if (ret)
  {
    debug(("IO_IOPeekUInt: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  
  *data = params.data;
  
  return(IO_RCC_SUCCESS);
}


/**********************************************************/
IO_ErrorCode_t IO_IOPeekUShort(u_int address, u_short *data)
/**********************************************************/
{
  int ret;
  IO_RCC_IO_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  if (address & 0x1)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));
  
  params.offset = address;
  params.size = 2;

  ret = ioctl(fd, IOPEEK, &params);
  if (ret)
  {
    debug(("IO_IOPeekUShort: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  
  *data = (u_short)params.data;
  
  return(IO_RCC_SUCCESS);
}


/********************************************************/
IO_ErrorCode_t IO_IOPeekUChar(u_int address, u_char *data)
/********************************************************/
{
  int ret;
  IO_RCC_IO_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  params.offset = address;
  params.size = 1;
  
  ret = ioctl(fd, IOPEEK, &params);
  if (ret)
  {
    debug(("IO_IOPeekUChar: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  
  *data = (u_char)params.data;
  
  return(IO_RCC_SUCCESS);
}


/*****************************************************/
IO_ErrorCode_t IO_IOPokeUInt(u_int address, u_int data)
/*****************************************************/
{
  int ret;
  IO_RCC_IO_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));

  if (address & 0x3)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));

  params.offset = address;
  params.data = data;
  params.size = 4;
  
  ret = ioctl(fd, IOPOKE, &params);
  if (ret)
  {
    debug(("IO_IOPokeUInt: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  
  return(IO_RCC_SUCCESS);
}


/*********************************************************/
IO_ErrorCode_t IO_IOPokeUShort(u_int address, u_short data)
/*********************************************************/
{
  int ret;
  IO_RCC_IO_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));

  if (address & 0x1)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));

  params.offset = address;
  params.data = data;
  params.size = 2;

  ret = ioctl(fd, IOPOKE, &params);
  if (ret)
  {
    debug(("IO_IOPokeUShort: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  
  return(IO_RCC_SUCCESS);
}


/*******************************************************/
IO_ErrorCode_t IO_IOPokeUChar(u_int address, u_char data)
/*******************************************************/
{
  int ret;
  IO_RCC_IO_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));

  params.offset = address;
  params.data = data;
  params.size = 1;
  
  ret = ioctl(fd, IOPOKE, &params);
  if (ret)
  {
    debug(("IO_IOPokeUChar: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  
  return(IO_RCC_SUCCESS);
}


/************************************************************************************************/
IO_ErrorCode_t IO_PCIDeviceLink(u_int vendor_id, u_int device_id, u_int occurrence, u_int *handle)
/************************************************************************************************/
{
  IO_PCI_FIND_t params;
  int ret;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  params.vid = vendor_id;
  params.did = device_id;
  params.occ = occurrence;
  
  printf("\nVID=0x%x\tDID=0x%x\tOCC=%d",params.vid,params.did,params.occ);

  ret = ioctl(fd, IOPCILINK, &params);
  if (ret)
  {
    debug(("IO_PCIDeviceLink: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }
  
  *handle = params.handle;
  return(IO_RCC_SUCCESS);
}


/*********************************************/
IO_ErrorCode_t IO_PCIDeviceUnlink(u_int handle)
/*********************************************/
{
  int ret;
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  ret = ioctl(fd, IOPCIUNLINK, &handle);
  if (ret)
  {
    debug(("IO_PCIDeviceUnlink: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }     
        
  return(IO_RCC_SUCCESS);
}


/**************************************************************************/
IO_ErrorCode_t IO_PCIConfigReadUInt(u_int handle, u_int offset, u_int *data)
/**************************************************************************/
{
  int ret;
  IO_PCI_CONF_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  if (offset & 0x3)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));
    
  params.handle = handle;
  params.offs = offset;
  params.size = 4;

  fprintf(stdout,"\nHandle=%d\tOffset=0x%8x\tSize=4\n",params.handle,params.offs);
  /*
  // temp
  ret = ioctl(fd, IOGETID, &params);
  if (ret)
  {
    debug(("IO_PCIConfigReadUInt: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }   
  *data = params.data;
  fprintf(stdout, "ID from ioctl: 0x%8x\n", *data );
  */
  //
  ret = ioctl(fd, IOPCICONFR, &params);
  if (ret)
  {
    debug(("IO_PCIConfigReadUInt: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }   

  *data = params.data;
  fprintf(stdout,"\nCONFR from ioctl: 0x%8x\n",*data);  

  return(IO_RCC_SUCCESS);
}


/******************************************************************************/
IO_ErrorCode_t IO_PCIConfigReadUShort(u_int handle, u_int offset, u_short *data)
/******************************************************************************/
{
  int ret;
  IO_PCI_CONF_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  if (offset & 0x1)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));
  
  params.handle = handle;
  params.offs = offset;
  params.size = 2;

  ret = ioctl(fd, IOPCICONFR, &params);
  if (ret)
  {
    debug(("IO_PCIConfigReadUShort: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }   
  
  *data = (u_short)params.data;

  return(IO_RCC_SUCCESS);
}


/****************************************************************************/
IO_ErrorCode_t IO_PCIConfigReadUChar(u_int handle, u_int offset, u_char *data)
/****************************************************************************/
{
  int ret;
  IO_PCI_CONF_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  params.handle = handle;
  params.offs = offset;
  params.size = 1;

  ret = ioctl(fd, IOPCICONFR, &params);
  if (ret)
  {
    debug(("IO_PCIConfigReadUChar: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }   
  
  *data = (u_char)params.data;

  return(IO_RCC_SUCCESS);
}


/**************************************************************************/
IO_ErrorCode_t IO_PCIConfigWriteUInt(u_int handle, u_int offset, u_int data)
/**************************************************************************/
{
  int ret;
  IO_PCI_CONF_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
    
  if (offset & 0x3)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));
    
  params.handle = handle;
  params.offs = offset;
  params.data = data;  
  params.size = 4;

  ret = ioctl(fd, IOPCICONFW, &params);
  if (ret)
  {
    debug(("IO_PCIConfigWriteUInt: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }   

  return(IO_RCC_SUCCESS);
}


/******************************************************************************/
IO_ErrorCode_t IO_PCIConfigWriteUShort(u_int handle, u_int offset, u_short data)
/******************************************************************************/
{
  int ret;
  IO_PCI_CONF_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
    
  if (offset & 0x1)
    return(RCC_ERROR_RETURN(0, IO_RCC_ILL_OFFSET));
    
  params.handle = handle;
  params.offs = offset;
  params.data = data;  
  params.size = 2;

  ret = ioctl(fd, IOPCICONFW, &params);
  if (ret)
  {
    debug(("IO_PCIConfigWriteUShort: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }   

  return(IO_RCC_SUCCESS);
}


/****************************************************************************/
IO_ErrorCode_t IO_PCIConfigWriteUChar(u_int handle, u_int offset, u_char data)
/****************************************************************************/
{
  int ret;
  IO_PCI_CONF_t params;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
  
  params.handle = handle;
  params.offs = offset;
  params.data = data;  
  params.size = 1;

  ret = ioctl(fd, IOPCICONFW, &params);
  if (ret)
  {
    debug(("IO_PCIConfigWriteUChar: Error from ioctl, errno=%d\n", errno));
    return(RCC_ERROR_RETURN(0, errno));
  }   

  return(IO_RCC_SUCCESS);
}


/**************************************************/
IO_ErrorCode_t IO_GetHostInfo(HostInfo_t *host_info)
/**************************************************/
{
  int ret;
  u_char d1;
  
  if (!is_open) 
    return(RCC_ERROR_RETURN(0, IO_RCC_NOTOPEN));
    
  // Read the board identification
  ret = IO_IOPokeUChar(CMOSA, BID1);
  if (ret != IO_RCC_SUCCESS)
    return(RCC_ERROR_RETURN(0, IO_RCC_IOFAIL));
 
  ret = IO_IOPeekUChar(CMOSD, &d1);
  if (ret != IO_RCC_SUCCESS)
    return(RCC_ERROR_RETURN(0, IO_RCC_IOFAIL));

  if (!(d1 & 0x80))
    return(RCC_ERROR_RETURN(0, IO_RCC_ILLMANUF));
  
  ret = IO_IOPokeUChar(CMOSA, BID2);
  if (ret != IO_RCC_SUCCESS)
    return(RCC_ERROR_RETURN(0, IO_RCC_IOFAIL));

  ret = IO_IOPeekUChar(CMOSD, &d1);
  if (ret != IO_RCC_SUCCESS)
    return(RCC_ERROR_RETURN(0, IO_RCC_IOFAIL));


  d1 &= 0x1f;  // Mask board ID bits
  host_info->board_type = VP_UNKNOWN;
       if (d1 == 2) host_info->board_type = VP_PSE;
  else if (d1 == 3) host_info->board_type = VP_PSE;
  else if (d1 == 4) host_info->board_type = VP_PSE;
  else if (d1 == 5) host_info->board_type = VP_PMC;
  else if (d1 == 6) host_info->board_type = VP_CP1;
  else if (d1 == 7) host_info->board_type = VP_100;
  
  host_info->board_manufacturer = CCT;
  host_info->operating_system_type = LINUX;
  host_info->operating_system_version = K249;  

  return(IO_RCC_SUCCESS);
}


/******************************************************************/
unsigned int IO_RCC_err_get(err_pack err, err_str pid, err_str code)
/******************************************************************/
{ 
  strcpy(pid, P_ID_IO_RCC_STR);

  switch (RCC_ERROR_MINOR(err))
  {  
    case IO_RCC_SUCCESS:       strcpy(code, IO_RCC_SUCCESS_STR);         break;
    case IO_RCC_FILE:          strcpy(code, IO_RCC_FILE_STR);            break;
    case IO_RCC_NOTOPEN:       strcpy(code, IO_RCC_NOTOPEN_STR);         break;
    case IO_RCC_MMAP:          strcpy(code, IO_RCC_MMAP_STR);            break;
    case IO_RCC_MUNMAP :       strcpy(code, IO_RCC_MUNMAP_STR);          break;
    case IO_RCC_ILLMANUF:      strcpy(code, IO_RCC_ILLMANUF_STR);        break;
    case IO_RCC_IOFAIL:        strcpy(code, IO_RCC_IOFAIL_STR);          break;
    case IO_PCI_TABLEFULL:     strcpy(code, IO_PCI_TABLEFULL_STR);       break;
    case IO_PCI_NOT_FOUND:     strcpy(code, IO_PCI_NOT_FOUND_STR);       break;
    case IO_PCI_ILL_HANDLE:    strcpy(code, IO_PCI_ILL_HANDLE_STR);      break;
    case IO_PCI_UNKNOWN_BOARD: strcpy(code, IO_PCI_UNKNOWN_BOARD_STR);   break;
    case IO_PCI_CONFIGRW:      strcpy(code, IO_PCI_CONFIGRW_STR);        break;
    case IO_PCI_REMAP:         strcpy(code, IO_PCI_REMAP_STR);           break;
    case IO_RCC_ILL_OFFSET:    strcpy(code, IO_RCC_ILL_OFFSET_STR);      break;
    default:                   strcpy(code, IO_RCC_NO_CODE_STR); return(RCC_ERROR_RETURN(0,IO_RCC_NO_CODE)); break;
  }
  return(RCC_ERROR_RETURN(0, IO_RCC_SUCCESS));
}


