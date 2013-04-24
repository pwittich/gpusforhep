/************************************************************************/
/*									*/
/*  This is the application  header file for the IO_RCC 		*/
/*  library & applications						*/
/*									*/
/*   6. Jun. 02  MAJO  created						*/
/*									*/
/*******C 2002 - The software with that certain something****************/

#ifndef _IO_RCC_H
#define _IO_RCC_H

#include "io_rcc_common.h"

#ifdef CMEM_RCC_LIB_DEBUG
  #define debug(x) printf x
#else
  #define debug(x)
#endif


//Definitions used in IO_GetHostInfo
//Board type definitions are in io_rcc_common.h

//Board manufacturers
#define CCT 1

//Operating system type
#define LINUX 1

//Operating system version
#define K249 1

//Error strings
#define IO_RCC_SUCCESS_STR       "No error"
#define IO_RCC_ERROR_FAIL_STR    "Failed to install the error library"
#define IO_RCC_FILE_STR          "Failed to open /dev/io_rcc"
#define IO_RCC_NOTOPEN_STR       "Library has not yet been opened"
#define IO_RCC_MMAP_STR          "Error from call to mmap function"
#define IO_RCC_MUNMAP_STR        "Error from call to munmap function"
#define IO_RCC_ILLMANUF_STR      "Unable to determine board manufacturer"
#define IO_RCC_IOFAIL_STR        "Error from IO_IOPeek or IO_IOPoke"
#define IO_RCC_NO_CODE_STR       "Unknown error"
#define IO_PCI_TABLEFULL_STR     "Internal device table is full"
#define IO_PCI_NOT_FOUND_STR     "PCI Device not found"
#define IO_PCI_ILL_HANDLE_STR    "Illegal handle"
#define IO_PCI_CONFIGRW_STR      "Error from pci_(read/write)_config_dword system call"
#define IO_PCI_UNKNOWN_BOARD_STR "Board type can not be determined"
#define IO_RCC_ILL_OFFSET_STR    "Illegal offset (alignment)"
#define IO_PCI_REMAP_STR         "Error from remap_page_range system call"

#ifdef __cplusplus
extern "C" {
#endif
IO_ErrorCode_t IO_Open(void);
IO_ErrorCode_t IO_Close(void);
IO_ErrorCode_t IO_PCIMemMap(u_int pci_addr, u_int size, u_int *virt_addr);
IO_ErrorCode_t IO_PCIMemUnmap(u_int virt_addr, u_int size);
IO_ErrorCode_t IO_IOPeekUInt(u_int address, u_int *data);
IO_ErrorCode_t IO_IOPokeUInt(u_int address, u_int data);
IO_ErrorCode_t IO_IOPeekUShort(u_int address, u_short *data);
IO_ErrorCode_t IO_IOPokeUShort(u_int address, u_short data);
IO_ErrorCode_t IO_IOPeekUChar(u_int address, u_char *data);
IO_ErrorCode_t IO_IOPokeUChar(u_int address, u_char data);
IO_ErrorCode_t IO_PCIDeviceLink(u_int vendor_id, u_int device_id, u_int occurrence, u_int *handle);
IO_ErrorCode_t IO_PCIDeviceUnlink(u_int handle);
IO_ErrorCode_t IO_PCIConfigReadUInt(u_int handle, u_int offset, u_int *data);
IO_ErrorCode_t IO_PCIConfigWriteUInt(u_int handle, u_int offset, u_int data);
IO_ErrorCode_t IO_PCIConfigReadUShort(u_int handle, u_int offset, u_short *data);
IO_ErrorCode_t IO_PCIConfigWriteUShort(u_int handle, u_int offset, u_short data);
IO_ErrorCode_t IO_PCIConfigReadUChar(u_int handle, u_int offset, u_char *data);
IO_ErrorCode_t IO_PCIConfigWriteUChar(u_int handle, u_int offset, u_char data);
IO_ErrorCode_t IO_GetHostInfo(HostInfo_t *host_info);
unsigned int IO_RCC_err_get(err_pack err, err_str pid, err_str code);
#ifdef __cplusplus
}
#endif

#endif
