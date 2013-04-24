/************************************************************************/
/*									*/
/*  This is the driver header file for the IO_RCC package		*/
/*									*/
/*   6. Jun. 02  MAJO  created						*/
/*									*/
/*******C 2002 - The software with that certain something****************/

#ifndef _IO_RCC_DRIVER_H
#define _IO_RCC_DRIVER_H

#include "io_rcc_common.h"

#ifdef IO_RCC_DEBUG
  #define kdebug(x) printk x
#else
  #define kdebug(x)
#endif

#define IO_MAX_PCI       100 //Max. number of PCI devices linked at any time

#define CMOSA            0x70
#define CMOSD            0x71
#define BID1             0x35
#define BID2             0x36

typedef struct
{
  struct pci_dev *dev_ptr;
  u_int vid;
  u_int did;
  u_int occ;
  u_int pid;
} pci_devices_t;

/*************/
/*ioctl codes*/
/*************/
enum
{
  IOPEEK=1,
  IOPOKE,
  IOGETID,
  IOPCILINK,
  IOPCIUNLINK,
  IOPCICONFR,
  IOPCICONFW
};


#endif
