
#include <iostream.h>
#include <fstream.h>
#include <time.h>
#include <string.h>
#include <sys/io.h>
#include <stdarg.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>

// #ifdef TSTAMP
// #include "TROOT.h"
// #include "TNtuple.h"
// #include "TFile.h"
// #endif

#include "rcc_error.h"
#include "io_rcc.h"
#include "cmem_rcc.h"
#include "solar_map.h"
#include "L2Types.hh"

#ifdef SFIO
#include <unistd.h>
#include <linux/types.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <sched.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>
#include "sfio.h"
#endif

extern "C"{
#include "solarc.h"
#include "io_rcc.h"
#include "rcc_error.h"
}

/***********
 * Globals *
 ***********/
#define CHANNELS 2		// we will not use channel[0] 
#define MAX_EVENTS   59000
#define BLOCK_WORDS  48
#define NUM_WORDS    576

#define SCW 0xB0F00000
#define ECW 0xE0F00000
#ifndef S_BUFSIZE
#define S_BUFSIZE  ( 4*576 )      // bytes for 1 event
#endif
#define OCCS 1
#define OCCF 1
#define MAXBUF 1
#define MAXREQ 1
#define PAGESIZE 4					    /* Corresponds to 16kB */
#define PREFILL 0xdeadbeef
#define CLOCKS_PER_USEC 1396.449   // pcpulsar
//#define CLOCKS_PER_USEC 2399.567   // pcpulsar2



// solar stuff
extern int shandle[MAXBUF];
extern unsigned int  paddr_s[MAXBUF], uaddr_s[MAXBUF];
extern volatile solar_regs_t *solar;
extern unsigned int solar_regs, solar_handle;



/*************
 * map_solar *
 *************/

void map_solar(void) {
  unsigned int uret;
  unsigned int pciaddr_s, offset;

  uret = IO_PCIDeviceLink(0x10dc, 0x0017, OCCS, &solar_handle);
  if (uret != IO_RCC_SUCCESS) rcc_die(uret, "IO_PCIDeviceLink(solar)");
  uret = IO_PCIConfigReadUInt(solar_handle, 0x10, &pciaddr_s);
  if (uret != IO_RCC_SUCCESS) rcc_die(uret, "IO_PCIConfigReadUInt(solar)");
  offset = pciaddr_s & 0xfff;
  pciaddr_s &= 0xfffff000;
  uret = IO_PCIMemMap(pciaddr_s, 0x1000, &solar_regs);
  if (uret != IO_RCC_SUCCESS) rcc_die(uret, "IOPCIMemMap(solar)");

  solar = (solar_regs_t *)(solar_regs + offset);

}




/********************
 * solar_card_reset *
 ********************/

void solar_card_reset(void) {
  unsigned int data;

  data = solar->opctrl;
  data |= 0x1;
  solar->opctrl = data;
  sleep(1);
  data &= 0xfffffffe;
  solar->opctrl = data;

}

/********************
 * solar_link_reset *
 ********************/

void solar_link_reset(void) {
  unsigned int status;

  /* clear the URESET bit to make sure that there is a falling edge on URESET_N */
  solar->opctrl &= 0xfffdffff;
  /*set the URESET bits*/
  solar->opctrl |= 0x00020000;
  sleep(1); /* wait to give the link time to come up */

  /*now wait for LDOWN to come up again*/
  printf("Waiting for link to come up...\n");
  while((status = solar->opstat) & 0x00020000)
    printf("solar->opstat = 0x%08x\n", status);

  /*reset the URESET bits*/
  solar->opctrl &= 0xfffdffff;
}


/***********
 * rcc_die *
 ***********/

void rcc_die(unsigned int code, const char *s, ...) {
  int i;
  va_list ap;

  va_start(ap, s);
  vfprintf(stderr, s, ap);
  fprintf(stderr, ":\n");
  rcc_error_print(stderr, code);
  
  IO_Close();
  for (i=0;i<MAXBUF;i++)
    CMEM_SegmentFree(shandle[i]);
/*   CMEM_SegmentFree(shandle); */
  CMEM_Close();
  
  va_end(ap);
  exit(code);
}

/*******
 * die *
 *******/

void die(const char *s, ...) {
  va_list ap;
  
  va_start(ap, s);
  vfprintf(stderr, s, ap);
  fprintf(stderr, ": ");
  perror(NULL);
  
  va_end(ap);
  exit(1);
}

/***************
 * unmap_solar *
 ***************/

void unmap_solar(void) {
  unsigned int uret;

  uret = IO_PCIMemUnmap(solar_regs, 0x1000);
  if (uret)
    rcc_error_print(stderr, uret);

}


/***********
 * chkconw *
 ***********/

int chkconw(unsigned long long loop, unsigned int data, unsigned int data2,
	     unsigned int data3, unsigned int fsize, unsigned int expect_size) {
  int errz = 0;
  
  if (data & 0x80000000) {
    printf("Packet %llu: Start control word MISSING!\n", loop);
    errz++;
  }
  if (data2 != SCW) {
    printf("Packet %llu: MISMATCH in start control word!\n", loop);
    errz++;
  }
  
  if (data & 0x20000000) {
    printf("Packet %llu: End control word MISSING!\n", loop);
    errz++;
  }
  if (data3 != ECW) {
    printf("Packet %llu: MISMATCH in end control word!\n", loop);
    errz++;
  }
  
  if (fsize != expect_size) {
    printf("Packet %llu: Packet size did not match expected size! (%u != %u)\n", 
	   loop, fsize, expect_size);
    errz++;
  }
  
  /* Check the content of the ACK FIFO */
  if (data2 & 0x3) {
    printf("Packet %llu: Error %d in start control word\n",
	   loop, data2 & 0x3);
  }
  if (data3 & 0x3) {
    printf("Packet %llu: Error %d in end control word\n",
	   loop, data3 & 0x3);
  }
  
  return(errz);
}




