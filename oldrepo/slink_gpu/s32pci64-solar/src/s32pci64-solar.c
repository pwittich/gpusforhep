
/* ----------------------------------------------------------------- */
/* solarc.c                                                          */
/*                                                                   */
/* repackaged CERN interface code that supports multiple solars      */
/*                                                                   */
/* KH - 4.9.04                                                       */
/* ----------------------------------------------------------------- */


/******************
      Headers    
******************/
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "rcc_error/rcc_error.h"
#include "io_rcc/io_rcc.h"
#include "cmem_rcc/cmem_rcc.h"

#include "s32pci64-solar/s32pci64-solar.h"
#include "s32pci64-solar/solar_map.h"

#include "s32pci64-solar/s32pci64-cdf.h" 

/******************
    Definitions 
******************/
#define PAGESIZE            4	    /* Corresponds to 16kB */
#define S_BUFSIZE           1024
#define MAXSOLARS           2
//#define MAXBUF              1
#define MAXBUF              4

/**************
    Globals 
**************/
static unsigned int data[MAXSOLARS][MAXBUF][S_BUFSIZE];
static int bhandle[MAXSOLARS][MAXBUF];
static unsigned int  paddr[MAXSOLARS][MAXBUF];
static unsigned int  uaddr[MAXSOLARS][MAXBUF];
static volatile T_solar_regs *solar[MAXSOLARS];
static unsigned int solar_regs[MAXSOLARS], solar_handle[MAXSOLARS];




/***********************/
int solar_map(int occ) {
/***********************/
  unsigned int eret;
  unsigned int pciaddr, offset;

  eret = IO_Open();
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

  /* KH  occ+1 ==  PCI<->array translation */
  eret = IO_PCIDeviceLink(0x10dc, 0x0017, occ+1, &solar_handle[occ]);
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

  eret = IO_PCIConfigReadUInt(solar_handle[occ], 0x10, &pciaddr);
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

  offset = pciaddr & 0xfff;
  pciaddr &= 0xfffff000;
  eret = IO_PCIMemMap(pciaddr, 0x1000, &solar_regs[occ]);
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

  solar[occ] = (T_solar_regs  *)(solar_regs[occ] + offset);

  return (0);
}


/************************/
int solar_exit(int occ)
/************************/
{
  unsigned int loop, eret;
  
  for (loop = 0; loop < MAXBUF; loop++) 
    {
      eret = CMEM_SegmentFree(bhandle[occ][loop]);
      if (eret) {
	printf("Warning: Failed to free buffer #%d for filar %d\n",
	       loop + 1, occ );
	rcc_error_print(stdout, eret);
      }
    }

  eret = CMEM_Close();
  if (eret) 
    {
      fprintf( stderr, "Warning: Failed to close the CMEM_RCC library for solar %d\n", occ);
      rcc_error_print(stdout, eret);
    }

  IO_Close();

  return (0);
  
}



/************************/
int solar_unmap(int occ)
/************************/
{
  unsigned int eret;

  fprintf( stderr, "unmapping solar\n");

  eret = IO_PCIMemUnmap(solar_regs[occ], 0x1000);
  if (eret)
    rcc_error_print(stderr, eret);

  eret = IO_Close();
  if (eret)
    rcc_error_print(stdout, eret);
  return (0);
}



/***********************/
int solar_init(int occ)
/***********************/
{
  unsigned int chan, *ptr, loop2, loop, eret;
  char cmem_string[128];

  eret = CMEM_Open();
  if (eret) {
    printf("Sorry. Failed to open the cmem_rcc library for solar %d \n", occ);
    rcc_error_print(stdout, eret);
    exit(6);
  }
  
  sprintf( cmem_string, "solar%d", occ );
  printf( "CMEM:  S_BUFSIZE set to %d\n", S_BUFSIZE );

    
  for (loop = 0; loop < MAXBUF; loop++) {

    eret = CMEM_SegmentAllocate(S_BUFSIZE, cmem_string, &bhandle[occ][loop]);
      if (eret) {
	printf("Sorry. Failed to allocate buffer #%d for solar %d\n",
	       occ, loop + 1 );
	rcc_error_print(stdout, eret);
	exit(7);
      }
      
      eret =
	CMEM_SegmentVirtualAddress(bhandle[occ][loop],
				   &uaddr[occ][loop]);
      if (eret) {
	printf("Sorry. Failed to get virtual address for solar %d buffer #%d\n",
	       occ, loop + 1 );
	rcc_error_print(stdout, eret);
	exit(8);
      }
      
      eret =
	CMEM_SegmentPhysicalAddress(bhandle[occ][loop],
				    &paddr[occ][loop]);
      if (eret) {
	printf("Sorry. Failed to get physical address for solar %d buffer #%d\n ",
	        occ, loop + 1 );
	rcc_error_print(stdout, eret);
	exit(9);
      }
      
      /*** initialise the buffer ***/
      ptr = (unsigned int *) uaddr[occ][loop];
      /*       for (loop2 = 0; loop2 < (S_BUFSIZE >> 2); loop2++) */
      for (loop2 = 0; loop2 < (S_BUFSIZE / PAGESIZE ); loop2++)
	*ptr++ = PREFILL;

    } // buffer loop
}


/************************/
void solar_setup(int occ) 
/************************/
{ 

  unsigned int eret, data;

  solar_init(occ);
  solar_map(occ);
  
  eret = IO_PCIConfigReadUInt(solar_handle[occ], 0x8, &data);
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }
  
  fprintf( stderr, "solar_cardreset(%d)\n", occ  );  
  solar_cardreset(occ);
  fprintf( stderr, "solar_linkreset(%d)\n", occ  );  
  solar_linkreset(occ);

  /*** set start and end ctrl words ***/
  solar[occ]->bctrlw = SCW & 0xfffffffc;
  solar[occ]->ectrlw = ECW & 0xfffffffc;
  
  /* solar[occ]->opctrl = (PAGESIZE << 3) */;       /* 16kB Page size */

}



/***************************/
int solar_cardreset(int occ) 
/***************************/
{

  unsigned int data;

  data = solar[occ]->opctrl;
  data |= 0x1;
  solar[occ]->opctrl = data;
  sleep(1);
  data &= 0xfffffffe;
  solar[occ]->opctrl = data;

  return (0);

}

/***************************/
int solar_linkreset(int occ)
/***************************/
{

  unsigned int status;

  /*** clear the URESET bit to make sure that ***/
  /***  there is a falling edge on URESET_N   ***/
  solar[occ]->opctrl &= 0xfffdffff;

  /*** set the URESET bits ***/
  solar[occ]->opctrl |= 0x00020000;
  /*** wait to give the link time to come up ***/
  sleep(1); 

  /*** now wait for LDOWN to come up again ***/
  printf("Waiting for Solar link to come up...\n");
  while((status = solar[occ]->opstat) & 0x00020000)
    printf("solar[%d]->opstat = 0x%08x\n", occ, status);

  printf("Solar link is up!\n");

  /*** reset the URESET bits ***/
  solar[occ]->opctrl &= 0xfffdffff;

  return (0);
}


/*****************************************/
unsigned int * solar_getbuffer(int occ)
/*****************************************/
{
  // what about the MAXBUF index?
  return &data[occ][0][0];
}

/***********************/
int solar_send( int len , int occ)
/***********************/
{

  int q, sfree, fsize = 0;
  unsigned int  *ptr;
  // count=0;
  //  static unsigned int expect_size = 126;

  /*** setup ***/
  /* expect_size = S_BUFSIZE/4; */

/*   if( len > S_BUFSIZE/4 )  */
/*     fprintf( stderr, "transmission length > S_BUFSIZE/4\n" );  */

  /* !!! this doesn't seem to work anymore  !!! */
  /* !!! maybe we have latent entries, find !!! */
  /* !!! a way to get rid of them on setup  !!! */
  /*   sfree = solar[occ]->opstat & 0xf; */
  /*   if (sfree == 0) { */
  /*     fprintf(stderr, "Out of space in solar (%d) req fifo! Exiting...\n", occ ); */
  /*     return (1); */
  /*   }     */

  ptr =  (unsigned int *)uaddr[occ][0];

  /***  read in the data ***/
  for( q=0; q<len; q++ )
    {
      *ptr = (unsigned long int)( *(data[occ][0]+q) & 0xffffffff );
      ptr++;
    }

  /***  send the decision ***/
  /* fsize = S_BUFSIZE/4; */
  fsize = len;
  solar[occ]->reqfifo1 = paddr[occ][0];
  solar[occ]->reqfifo2 = (1 << 31) + (1 << 30) + fsize;
  
  /*** wait until ready ***/
  while (1) 
    {
      sfree = solar[occ]->opstat & 0xf;
      if (sfree == MAXREQ)
	break;
      /* count++; */
    }

}

/***********************/
int solar_send_ptr( int len, unsigned* userptr, int occ)
/***********************/
{
  int q, sfree, fsize = 0;
  unsigned int  *ptr;
  int count = 0 ;


  /*** check req fifo ***/
  /*
  sfree = solar[occ]->opstat & 0xf;
  if (sfree != MAXREQ)
    return 1;
  */

  ptr =  (unsigned int *)uaddr[occ][0];

  /***  read in the data ***/
  for( q=0; q<len; q++ )
    {
      *ptr = (unsigned long)( *(userptr+q) & 0xffffffff );
      ptr++;
    }

  /***  send the decision ***/
  fsize = len;
  solar[occ]->reqfifo1 = paddr[occ][0];
  solar[occ]->reqfifo2 = (1 << 31) + (1 << 30) + fsize;  

  /*** wait until ready ***/
  /*
  while (count < 200000) 
    {
      sfree = solar[occ]->opstat & 0xf;
      if (sfree == MAXREQ)
	break;
       count++; 
    }
  if( count>0 )
    fprintf( stderr, "broke loop after %d tries\n", count ); 
  */

  return 0;

}


/*********************************************************************/
int solar_send_ptr4( int len, unsigned* userptr, int occ, int buffer)
/*********************************************************************/
{
  int q, sfree, fsize = 0;
  unsigned int  *ptr;
  int count = 0 ;


  /*** check req fifo ***/
  /*
  sfree = solar[occ]->opstat & 0xf;
  if (sfree != MAXREQ)
    return 1;
  */

  ptr =  (unsigned int *)uaddr[occ][buffer];

  /***  read in the data ***/
  for( q=0; q<len; q++ )
    {
      *ptr = (unsigned long)( *(userptr+q) & 0xffffffff );
      ptr++;
    }

  /***  send the decision ***/
  fsize = len;
  solar[occ]->reqfifo1 = paddr[occ][buffer];
  solar[occ]->reqfifo2 = (1 << 31) + (1 << 30) + fsize;  

  /*** wait until ready ***/
  /*
  while (count < 200000) 
    {
      sfree = solar[occ]->opstat & 0xf;
      if (sfree == MAXREQ)
	break;
       count++; 
    }
  if( count>0 )
    fprintf( stderr, "broke loop after %d tries\n", count ); 
  */

  return 0;

}

/************************/
void solar_reset(int occ) 
/************************/
{ 

  unsigned int eret, data;
  int clksel, clkdiv;

  solar_dump_opstat(0); 
  solar_dump_opctrl(0);

  // turn the test mode off. sometimes, the SOLAR gets into
  // test mode. card and link reset does not solve that
  // this will turn that register to 0 "by hand"
  solar_testmode_off(0);

  fprintf( stderr, "\n\n--solar_cardreset(%d)\n", occ  );  
  solar_cardreset(occ);
  fprintf( stderr, "--solar_linkreset(%d)\n", occ  );  
  solar_linkreset(occ);

  solar_dump_opstat(0); 
  solar_dump_opctrl(0);


  /*** set start and end ctrl words ***/
  solar[occ]->bctrlw = SCW & 0xfffffffc;
  solar[occ]->ectrlw = ECW & 0xfffffffc;
  
  /*** 40MHz clock, no division ***/
  /*
  clksel = 1;  clkdiv = 0;
  solar[occ]->opfeat = (clksel << 31) + (clkdiv << 29);  
  */
}

/************************/
void solar_testmode_on(int occ) 
/************************/
{ 
  unsigned int eret, data;
  data = solar[occ]->opctrl;
  solar[occ]->opctrl |= 0x00100000;
}

/************************/
void solar_testmode_off(int occ) 
/************************/
{ 
  unsigned int eret, data;
  data = solar[occ]->opctrl;
  solar[occ]->opctrl = 0x00000000;
}



/************************/
void solar_dump_opstat(int occ) 
/************************/
{ 
  unsigned int eret, data;
  data = solar[occ]->opstat;
  fprintf( stderr, "0x%x\n", data );
}

/************************/
void solar_dump_opctrl(int occ) 
/************************/
{ 
  unsigned int eret, data;
  data = solar[occ]->opctrl;
  fprintf( stderr, "0x%x\n", data );
}


/****************************/
int solar_req_free(int occ)
/****************************/
{
  int sfree;

  sfree = solar[occ]->opstat & 0xf;
  if (sfree != MAXREQ)
    return 0;
  return 1;
}
