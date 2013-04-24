/* ----------------------------------------------------------------- */
/* solarc.c                                                          */
/*                                                                   */
/* repackaged CERN interface code that supports multiple solars      */
/*                                                                   */
/* KH - 4.9.04                                                       */
/* ----------------------------------------------------------------- */


// FILAR firmware documentation:
// https://edms.cern.ch/file/337904/1/userguide.PDF
// FILAR driver software documentation:
// https://edms.cern.ch/file/356135/1/filar_doc.pdf
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
#include "rcc_time_stamp/tstamp.h"

#include "s32pci64-filar/s32pci64-filar.h"
#include "s32pci64-filar/filar_map.h"

#include "s32pci64-filar/s32pci64-cdf.h"


/****************
 * Definitions *
****************/
#ifndef F_BUFSIZE
#define F_BUFSIZE  2048	/*bytes */
#endif

#define MAXCHAR 128



/**************
    Globals 
**************/
/* user space buffer */
unsigned int evdata[MAXFILARS][CHANNELS][0x10000ULL];

/* buffer addresses and registers */
static unsigned int filar_handle[MAXFILARS];
static int bhandle[MAXFILARS][CHANNELS][MAXBUF]; 
static unsigned int paddr[MAXFILARS][CHANNELS][MAXBUF]; 
static unsigned int uaddr[MAXFILARS][CHANNELS][MAXBUF];
static int active[MAXFILARS][CHANNELS];
static int bfree[MAXFILARS][CHANNELS];  // do this elsewhere  = { 0, MAXBUF, MAXBUF, MAXBUF, MAXBUF };
static int nextbuf[MAXFILARS][CHANNELS]; // do this elsewhere = { 0, 0, 0, 0, 0 };
static unsigned int filar_regs[MAXFILARS], offset;
static volatile T_filar_regs *filar[MAXFILARS];

/* other globals */
static int  ret, fun, occ;
static unsigned int *ptr, data, data2, data3, eret;
static unsigned int cont, first_event;


/*****************************/
void SigQuitHandler(int signum)
/*****************************/
{
  cont = 0;
  debug(("SigQuitHandler: ctrl+// received\n"));
}


/********************/
int filar_map(int occ)
/********************/
{
  unsigned int eret, regbase, pciaddr;

  eret = IO_Open();
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

  /* KH  occ+1 ==  PCI<->array translation */
  eret = IO_PCIDeviceLink(0x10dc, 0x0014, occ+1, &filar_handle[occ]);
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

  printf("occ=%d\tfilar_handle[occ]=0x%x\n",occ,filar_handle[occ]);

  eret = IO_PCIConfigReadUInt(filar_handle[occ], 0x10, &pciaddr);
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

/*   offset = pciaddr & 0xfff; */
/*   pciaddr &= 0xfffff000; */
  printf("pciaddr=0x%x\tsize=0x100\tfilar_regs[occ]=0x%x\n",pciaddr,&filar_regs[occ]);
  fprintf(stdout, "offset in s32pci: 0x%8x\n", offset);
  eret = IO_PCIMemMap(pciaddr, 0x1000, &filar_regs[occ]);
  if (eret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, eret);
    exit(-1);
  }

  filar[occ] = (T_filar_regs *) (filar_regs[occ] + offset);

  return (0);
}


/***********************/
int filar_unmap(int occ)
/***********************/
{
  unsigned int eret;

  fprintf( stderr, "unmapping filar\n");

  eret = IO_PCIMemUnmap(filar_regs[occ], 0x1000);
  if (eret)
    rcc_error_print(stderr, eret);

  eret = IO_Close();
  if (eret)
    rcc_error_print(stdout, eret);
  return (0);
}


/**************************************/
unsigned int * filar_getbuffer(int occ, int chan)
/**************************************/
{
  return &evdata[occ][chan][0];
}

/**************************************/
unsigned int * filar_getbuffer_4(int occ, int l2buff)
/**************************************/
{
  return &evdata[occ][l2buff+1][0];
}


/*************************************************/
int filar_setreq(int occ, int channel, int number)
/*************************************************/
{
  static int chan = 1, num = 1;
  int free, mode, bufnr, data, loop;
  char buff[MAXCHAR];

  if (!channel)
    mode = 1;
  else
    mode = 0;

  if (!channel) {
    printf("Enter the channel to be filled (1..%d) ", CHANNELS - 1);
    chan = strtol( fgets( buff, MAXCHAR, stdin ), NULL, 10 );
    //chan = getdecd(chan);
  }
  else
    chan = channel;

  if (!number) {
    printf("Enter the number of entries to be sent (1..15) ");
    num = strtol( fgets( buff, MAXCHAR, stdin ), NULL, 10 );
    //num = getdecd(num);
  }
  else
    num = number;

  // data = filar[occ]->fifostat;
  /*  free = (data >> (chan * 8 - 4)) & 0xf;

  if (free < num) {
    printf
	("The Request FIFO of channel %d has only space for %d entries\n",
	 chan, free);
    return (2);
  }
  -Sakari */

  for (loop = 0; loop < num; loop++) {
    bufnr = filar_getbuf(occ, chan, mode);
    if (bufnr < 0) {
      printf("All memory buffers are in use; filar = %d\n", occ);
      return (1);
    }

    if (chan == 1)
      filar[occ]->req1 = paddr[occ][chan][bufnr];
    if (chan == 2)
      filar[occ]->req2 = paddr[occ][chan][bufnr];
    if (chan == 3)
      filar[occ]->req3 = paddr[occ][chan][bufnr];
    if (chan == 4)
      filar[occ]->req4 = paddr[occ][chan][bufnr];

    if (mode) {
      printf("FIFO of filar %d channel %d filled with PCI address=0x%08x\n", occ, 
	     chan, paddr[occ][chan][bufnr]);
      printf("-->buffer number is %d\n", bufnr);
    }
  }
  return (0);
}





/***********************/
int filar_init(int occ)
/***********************/
{
  unsigned int chan, *ptr, loop2, loop, eret;
  char cmem_string[128];

  eret = CMEM_Open();
  if (eret) {
    printf("Sorry. Failed to open the cmem_rcc library for filar %d \n", occ);
    rcc_error_print(stdout, eret);
    exit(6);
  }

  sprintf( cmem_string, "filar%d", occ );
  printf( "CMEM:  F_BUFSIZE set to %d\n", F_BUFSIZE );

  for (chan = 1; chan < CHANNELS; chan++) {

    // initialize availiblity buffers
    nextbuf[occ][chan] = 0;
    bfree[occ][chan] = MAXBUF;
    
    for (loop = 0; loop < MAXBUF; loop++) {
            fprintf( stderr, "loop\n", loop );      
      eret = 
	CMEM_SegmentAllocate(F_BUFSIZE, cmem_string, &bhandle[occ][chan][loop]);
      if (eret) {
	printf("Sorry. Failed to allocate buffer #%d for filar %d channel %d\n",
	       occ, loop + 1, chan);
	rcc_error_print(stdout, eret);
	exit(7);
      }
          fprintf( stderr, "CMEM_SegmentAllocate %d, %d, %d\n", occ, chan ,loop );      
      eret =
	CMEM_SegmentVirtualAddress(bhandle[occ][chan][loop],
				   &uaddr[occ][chan][loop]);
      if (eret) {
	printf("Sorry. Failed to get virtual address for filar %d buffer #%d "
	       "for channel %d\n", occ, loop + 1, chan);
	rcc_error_print(stdout, eret);
	exit(8);
      }
            fprintf( stderr, "CMEM_SegmentVirtualAddress %d, %d, %d\n", occ, chan ,loop );      
      eret =
	CMEM_SegmentPhysicalAddress(bhandle[occ][chan][loop],
				    &paddr[occ][chan][loop]);
      if (eret) {
	printf("Sorry. Failed to get physical address for filar %d buffer #%d "
	       "for channel %d\n", occ, loop + 1, chan);
	rcc_error_print(stdout, eret);
	exit(9);
      }
      //      fprintf( stderr,"CMEM_SegmentPhysicalAddress %d, %d, %d\n", occ, chan ,loop );
      
      /*initialise the buffer */
            fprintf( stderr,"filar_init: before uaddr\n"  );
      ptr = (unsigned int *) uaddr[occ][chan][loop];
            fprintf( stderr,"filar_init: after uaddr\n"  );
/*       for (loop2 = 0; loop2 < (F_BUFSIZE >> 2); loop2++) */
/* 	*ptr++ = PREFILL; */
//      fprintf( stderr,"filar_init: after loop\n"  );
//      fprintf( stderr,"buffer initialized %d, %d, %d\n", occ, chan ,loop );

    } // buffer loop
  }   // channel loop

}


/**********************/
int filar_exit(int occ)
/**********************/
{
  unsigned int chan, loop, eret;
  
  for (chan = 1; chan < CHANNELS; chan++) {
    for (loop = 0; loop < MAXBUF; loop++) {
      
      eret = CMEM_SegmentFree(bhandle[occ][chan][loop]);
      if (eret) {
	printf("Warning: Failed to free buffer #%d for filar %d channel %d\n",
	       loop + 1, occ, chan);
	rcc_error_print(stdout, eret);
      }

    }
  }

  eret = CMEM_Close();
  if (eret) {
    printf("Warning: Failed to close the CMEM_RCC library for filar %d\n", occ);
    rcc_error_print(stdout, eret);
  }

}


/***********************************************/
int filar_getbuf(int occ, int channel, int mode)
/***********************************************/
{
  int bufnr;

  if (bfree[occ][channel] == 0)
    return (-1);
  else {
    bfree[occ][channel]--;
    bufnr = nextbuf[occ][channel];
    nextbuf[occ][channel]++;
    if (nextbuf[occ][channel] > (MAXBUF - 1))
      nextbuf[occ][channel] = 0;
      if (mode)
        printf("Filar %d, Chan %d, Buffer %d allocated\n", occ, channel, bufnr);
    return (bufnr);
  }
}


/***********************************************/
// KH
// the free buffers aren't necess zero-aligned.
// here we return the 'oldest' buffer wrt the
// nextbuf ( think: next - (max-free) plus wrap.  
// This lets us function like a FIFO
//
int filar_retbuf(int occ, int channel, int mode)
/***********************************************/
{
  int bufnr;

  bufnr = nextbuf[occ][channel] - MAXBUF + bfree[occ][channel];
  if (bufnr < 0)
    bufnr += MAXBUF;
  bfree[occ][channel]++;
  if (mode == 1)
    printf("Filar %d, Chan %d, Buffer %d returned\n", occ, channel, bufnr);
  return (bufnr);
}




/****************************/
int filar_cardreset(int occ)
/****************************/
{
  unsigned int chan, data;

  data = filar[occ]->ocr;
  data |= 0x1UL;
  filar[occ]->ocr = data;
  //  sleep(1);
  usleep(10000);
  data &= 0xfffffffeUL;
  filar[occ]->ocr = data;

  /*reset the buffers */
  for (chan = 1; chan < CHANNELS; chan++) {
    bfree[occ][chan] = MAXBUF;
    nextbuf[occ][chan] = 0;
  }
  return (0);
}

/****************************/
int filar_linkreset_hola(int occ, int chan)
/****************************/
{
  unsigned int data, linkdown;
  int count = 0;

  /*set the URESET bits */
  printf("filar[%d]->osr = 0x%08x\n", occ, filar[occ]->osr);
  printf("RESETTING CHANNEL %d\n",chan);
  filar[occ]->ocr |= 0x04104100U;
  //  ts_delay(10);			/* to be sure. 1 us should be enough */
  sleep(1);
  //usleep(10000);

  // KH

  printf("filar[%d]->osr = 0x%08x\n", occ, filar[occ]->osr);
  printf("filar[%d]->ocr = 0x%08x\n", occ, filar[occ]->ocr);
  printf("filar[%d]->imask = 0x%08x\n", occ, filar[occ]->imask);
  printf("filar[%d]->scw = 0x%08x\n", occ, filar[occ]->scw);
  printf("filar[%d]->ecw = 0x%08x\n", occ, filar[occ]->ecw);

  /*now wait URESET */
  printf("Waiting for Filar link to come up...\n");
  register status = filar[occ]->osr;	
  linkdown = 1;
  int print_it=1;
  while ( linkdown ){
    linkdown = ((status=filar[occ]->osr)>>(16+(chan*4)))&0x1;
    count++;
    if (print_it==1){
      
      printf("filar[%d]--->osr = 0x%08x, channel %d is down ... \n", occ, filar[occ]->osr,  chan);
      if(count==10) print_it=0;
    }
  }
  
  /*
  // KH
  linksdown = ~( 0xeeeeffff | (1<<(16+(chan*4))) );

  printf("Waiting for Filar link to come up...\n");
  register status = filar[occ]->osr;	
  //  while ((status=filar[occ]->osr) & 0x11110000){
  //  while (((status=filar[occ]->osr&0xffff0000) & linksdown) != linksdown ){
  while (((status=filar[occ]->osr&0x11110000) & linksdown) != linksdown ){
    count++;
    if (count < 1000){
      printf("filar[%d]--->osr = 0x%08x but want channel %d/0x%08x to be set \n", occ, status, chan, linksdown);
    }
  }
*/
  printf("Filar link (channel %d) is up!\nfilar[%d]->osr = 0x%08x\n", chan, occ, filar[occ]->osr);

  /*reset the URESET bits */
  filar[occ]->ocr &= 0xfbefbeff;
  return (0);
}

/****************************/
int filar_linkreset(int occ)
/****************************/
{
  unsigned int data;
  int count = 0;

  /*set the URESET bits */
  printf("filar[%d]->osr = 0x%08x\n", occ, filar[occ]->osr);
  filar[occ]->ocr |= 0x04104100U;
  //  ts_delay(10);			/* to be sure. 1 us should be enough */
  ts_delay(10);			/* to be sure. 1 us should be enough */

  /*now wait URESET */
  printf("Waiting for Filar link to come up...\n");
  // while (filar[occ]->osr & 0x11110000){
  // while (filar[occ]->osr & 0x00100000){
  while (filar[occ]->osr & 0x10000000){
    count++;
    if (count < 1000){
      printf("filar[%d]--->osr = 0x%08x\n", occ, filar[occ]->osr);
    }
  }

  printf("Filar link is up!\nfilar[%d]->osr = 0x%08x\n", occ, filar[occ]->osr);

  /*reset the URESET bits */
  filar[occ]->ocr &= 0xfbefbeff;
  return (0);
}

/************************/
void filar_setup_hola(int occ, int channel_mask) 
/************************/
{ 

  int chan = 0;

  filar_init(occ);
  filar_map(occ);
  
  ret = IO_PCIConfigReadUInt(filar_handle[occ], 0x8, &data);
  if (ret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, ret);
    exit(-1);
  }
  
  fprintf( stderr, "filar_cardreset(%d)\n", occ  );  
  filar_cardreset(occ);
  fprintf( stderr, "channel mask = 0x%08x\n", channel_mask  );  
  fprintf( stderr, "filar_linkreset(%d)\n", occ  );  
  for (chan = 0; chan < CHANNELS; chan++) {
   if( channel_mask & (1<<chan) ){
	printf("resetting channel %d\n",chan);
        filar_linkreset_hola(occ, chan);
   }
  }	
  filar[occ]->ocr = (4  << 3);       /* 16kB Page size */

  /*Fill the REQ FIFO of all active channels with one entry */
  //  for (chan = 1; chan < CHANNELS; chan++) {
    
    //if (active[occ][chan]) {
      
      /*Write one address to the request FIFO */
  //      eret = filar_setreq(occ, chan, 1);
	//      eret = filar_setreq(occ, 1, 4);
  //      if (eret) {
  //	printf("Error %d received from setreq for channel %d\n", eret,
  //	       1);
  //	return;
  //      }

      //}
  //  }
}

/************************/
void filar_setup(int occ) 
/************************/
{ 

  int chan = 0;

  filar_init(occ);
  filar_map(occ);
  
  ret = IO_PCIConfigReadUInt(filar_handle[occ], 0x8, &data);
  if (ret != IO_RCC_SUCCESS) {
    rcc_error_print(stdout, ret);
    exit(-1);
  }
  
  fprintf( stderr, "filar_cardreset(%d)\n", occ  );  
  filar_cardreset(occ);
  fprintf( stderr, "filar_linkreset(%d)\n", occ  );  
  filar_linkreset(occ);
  
  filar[occ]->ocr = (4  << 3);       /* 16kB Page size */

  /*Fill the REQ FIFO of all active channels with one entry */
  //  for (chan = 1; chan < CHANNELS; chan++) {
    
    //if (active[occ][chan]) {
      
      /*Write one address to the request FIFO */
  //      eret = filar_setreq(occ, chan, 1);
	//      eret = filar_setreq(occ, 1, 4);
  //      if (eret) {
  //	printf("Error %d received from setreq for channel %d\n", eret,
  //	       1);
  //	return;
  //      }

      //}
  //  }
}




/********************************************/
int filar_receive( int occ, int* filar_err )
/********************************************/
{

  long long int cnt = 0, trunc_cnt = 0;
  double *times;
  int *fiforeads;

  double x = 0, x2 = 0;
  double trunc_x = 0, trunc_x2 = 0;
  double avg, rms, first;
  unsigned long long int st, ed;
  double deltat;
  static unsigned int scw = 0xb0f00000, ecw = 0xe0f000000,
    sw1 = 2, nol = 1, nol_save = 10;
  unsigned int fsize, rmode, chan, isready, size[CHANNELS], ffrag,
      complete, ok = 0, bnum;

  static unsigned int expect_size = 126;
  int j, q, output = 0;
  int fragment,  z, err=0, loop=0;

  int ready_count = 0;
  unsigned long long int decision_mask[2];

  unsigned int *ptr, *ptr2, data, data2, data3, eret; // evdata[CHANNELS][0x10000];



  //
  // find the active channels
  //
  data = filar[occ]->ocr;
  active[occ][1] = (data & 0x00000200) ? 0 : 1;
  active[occ][2] = (data & 0x00008000) ? 0 : 1;
  active[occ][3] = (data & 0x00200000) ? 0 : 1;
  active[occ][4] = (data & 0x08000000) ? 0 : 1;
  expect_size = 2;


  //
  // Receive a complete, potentially fragmented packet from all 
  //  active channels
  //
  ffrag = 1;
  complete = fragment = 0;
  for (chan = 1; chan < CHANNELS; chan++)
    size[chan] = 0;
  
  /*Fill the REQ FIFO of all active channels with one entry */
  for (chan = 1; chan < CHANNELS; chan++) {

    if (active[occ][chan]) {

      /*Write one address to the request FIFO */
      eret = filar_setreq(occ, chan, 1);
      if (eret) {
	printf("Error %d received from setreq for channel %d\n", eret,
	       chan);
	return (1);
      }

    }
  }

  

  loop = 0; 
  while (!complete) 
    { //!complete

#ifdef DEBUG3
      printf( "---> receive loop: \t %d\n", loop ); 
#endif

      if ( fragment )
	{
	  /*Fill the REQ FIFO of all active channels with one entry */
	  for (chan = 1; chan < CHANNELS; chan++) 
	    {
	      if (active[occ][chan]) 
		{
		  /*Write one address to the request FIFO */
		  eret = filar_setreq(occ,chan, 1);
		  if (eret) 
		    {
		      printf("Error %d received from setreq for channel %d\n", eret,
			     chan);
		      return (1);
		    }
		}
	    }
	}
      
      
      
      //Wait for a fragment to arrive on all active channels 
      ready_count=0;
      while (1) {

	data = filar[occ]->fifostat;
	isready = 1;

	if (active[occ][1] && !(data & 0x0000000f))
	    isready = 0;
	if (isready)
	    break;
      }

      
      
      /* Read the ACK FIFOs */
      for (chan = 1; chan < CHANNELS; chan++) {
	if (active[occ][chan]) {
	  if (chan == 1)
	    data = filar[occ]->ack1;
	  if (chan == 2)
	    data = filar[occ]->ack2;
	  if (chan == 3)
	    data = filar[occ]->ack3;
	  if (chan == 4)
	    data = filar[occ]->ack4;
	  data2 = filar[occ]->scw;
	  data3 = filar[occ]->ecw;
	  fsize = data & 0xfffff;
	  
	  
	  if (!(data & 0x20000000)) {
	    complete = 1;
	  } else {
	    fragment = 1;
	    //printf( "Fragment!\n" );
	  }
	  
#ifdef DEBUG3
	  printf( "------------------------------\n");
	  printf( "frag SCW:\t 0x%08x\n", data2 );
	  printf( "frag ECW:\t 0x%08x\n", data3 );
	  printf( "fsize:\t %d\n", fsize );
	  printf( "------------------------------\n");
#endif
	  err = 0;
	  if( data & 0x80000000UL ) 
	    {
	      fprintf( stdout, "Channel %d Start Control word MISSING\n", chan);
	      *filar_err = 0x1;
	    }
	  if( data & 0x20000000UL ) 
	    {
	      fprintf( stdout, "Channel %d End Control word MISSING\n", chan );
	      *filar_err = 0x2;
	    }
	  if( data2 & 0xF ) 
	    {
	      fprintf( stderr, "Start Control Word Error: 0x%08x  (should be 0x%08x)\n", data2,  SCW );
	      err = 1;
	      *filar_err = 0x4;
	    }
	  if( data3 & 0xF )
	    {
	      fprintf( stderr, "End Control Word Error: 0x%08x  (should be 0x%08x)\n", data3,  ECW );
	      err = 1;
	      *filar_err = 0x8;
	    }

	  

	  ffrag = 0;
	  /*return the buffer and get a pointer to the data */
	  bnum = filar_retbuf(occ,chan, 0);
	  ptr = (unsigned int *) uaddr[occ][chan][bnum];
	  
	  // append frag data to end of user-space buffer
	  if (fsize) {		
	    memcpy( &evdata[occ][chan][size[chan]], ptr, fsize * 4);
	    size[chan] += fsize;
	  }
	  
#ifdef DEBUG3
	  ptr2 = (unsigned int *) uaddr[occ][chan][bnum];
	  if( fsize )
	    {
	      for( q=0; q<fsize; q++ )
		{
		  printf( "%lx\t", *ptr2 );
		  ptr2++;
		}
	    }
	  printf("\n");
#endif



	  loop++;

	  
	} // if channel active
      } // channel loop
      
#ifdef DEBUG3
      printf( "fragment %d\n", loop );
#endif

    } // while fragment

  
#ifdef DEBUG3
  printf( "fsize = %d\n\n", fsize );
#endif

  return  fsize;
}

/********************************************/
int filar_receive_nonblock( int occ, int* filar_err )
/********************************************/
{

  long long int cnt = 0, trunc_cnt = 0;
  double *times;
  int *fiforeads;

  double x = 0, x2 = 0;
  double trunc_x = 0, trunc_x2 = 0;
  double avg, rms, first;
  unsigned long long int st, ed;
  double deltat;
  static unsigned int scw = 0xb0f00000, ecw = 0xe0f00000,
    sw1 = 2, nol = 1, nol_save = 10;
  unsigned int fsize, rmode, chan, isready, size[CHANNELS], ffrag,
      complete, ok = 0, bnum;

  static unsigned int expect_size = 126;
  int j, q, output = 0;
  int fragment,  z, err=0, loop=0;

  int ready_count = 0;
  unsigned long long int decision_mask[2];

  unsigned int *ptr, *ptr2, data, data2, data3, eret; // evdata[CHANNELS][0x10000];



  //
  // find the active channels
  //
  data = filar[occ]->ocr;
  //  printf("filar[%d]->ocr = 0x%08x\n", occ, filar[occ]->ocr);
  active[occ][1] = (data & 0x00000200) ? 0 : 1;
  active[occ][2] = (data & 0x00008000) ? 0 : 1;
  active[occ][3] = (data & 0x00200000) ? 0 : 1;
  active[occ][4] = (data & 0x08000000) ? 0 : 1;
  expect_size = 2;

  //
  // Receive a complete, potentially fragmented packet from all 
  //  active channels
  //
  ffrag = 1;
  complete = fragment = 0;
  for (chan = 1; chan < CHANNELS; chan++)
    size[chan] = 0;
  

  //Wait for a fragment to arrive on all active channels 
  data = filar[occ]->fifostat;
  if( !(data & 0x0000000f) ) // active[occ][1] &&
    return 0;

  loop = 0; 
  while (!complete) 
    { //!complete

#ifdef DEBUG3
      printf( "---> receive loop: \t %d\n", loop ); 
#endif

      if ( fragment )
	{
	  /*Fill the REQ FIFO of all active channels with one entry */
	  for (chan = 1; chan < CHANNELS; chan++) 
	    {
	      if (active[occ][chan]) 
		{
		  /*Write one address to the request FIFO */
		  eret = filar_setreq(occ,chan, 1);
		  if (eret) 
		    {
		      printf("Error %d received from setreq for channel %d\n", eret,
			     chan);
		      return (1);
		    }
		}
	    }
	}
      
 
      /* Read the ACK FIFOs */
      for (chan = 1; chan < CHANNELS; chan++) {
	if (active[occ][chan]) {
	  if (chan == 1)
	    data = filar[occ]->ack1;
	  if (chan == 2)
	    data = filar[occ]->ack2;
	  if (chan == 3)
	    data = filar[occ]->ack3;
	  if (chan == 4)
	    data = filar[occ]->ack4;
	  data2 = filar[occ]->scw;
	  data3 = filar[occ]->ecw;
	  fsize = data & 0xfffff;
	  
	  if (!(data & 0x20000000)) {
	    complete = 1;
	  } else {
	    fragment = 1;
	    printf( "Fragment!\n" );
	  }

#ifdef DEBUG3
	  printf( "------------------------------\n");
	  printf( "frag SCW:\t 0x%08x\n", data2 );
	  printf( "frag ECW:\t 0x%08x\n", data3 );
	  printf( "fsize:\t %d\n", fsize );
	  printf( "------------------------------\n");
#endif
	  err = 0;
	  if( data & 0x80000000UL ) 
	    {
	      fprintf( stdout, "Channel %d Start Control word MISSING\n", chan);
	      *filar_err = 0x1;
	    }
	  if( data & 0x20000000UL ) 
	    {
	      fprintf( stdout, "Channel %d End Control word MISSING\n", chan );
	      *filar_err = 0x2;
	    }
	  if( data2 & 0xF ) 
	    {
	      fprintf( stderr, "Channel %d Start Control Word Error: 0x%08x  (should be 0x%08x)\n", chan, data2,  SCW );
	      err = 1;
	      *filar_err = 0x4;
	    }
	  if( data3 & 0xF )
	    {
	      fprintf( stderr, "Channel %d End Control Word Error: 0x%08x  (should be 0x%08x)\n", chan, data3,  ECW );
	      err = 1;
	      *filar_err = 0x8;
	    }

	  

	  ffrag = 0;
	  /*return the buffer and get a pointer to the data */
	  bnum = filar_retbuf(occ,chan, 0);
	  ptr = (unsigned int *) uaddr[occ][chan][bnum];
	  
	  // append frag data to end of user-space buffer
	  if (fsize) {		
	    memcpy( &evdata[occ][chan][size[chan]], ptr, fsize * 4);
	    size[chan] += fsize;
	  }
	  
#ifdef DEBUG3
	  ptr2 = (unsigned int *) uaddr[occ][chan][bnum];
	  if( fsize )
	    {
	      for( q=0; q<fsize; q++ )
		{
		  printf( "%lx\t", *ptr2 );
		  ptr2++;
		}
	    }
	  printf("\n");
#endif



	  loop++;

	  
	} // if channel active
      } // channel loop
      
#ifdef DEBUG3
      printf( "fragment %d\n", loop );
#endif

    } // while fragment

  /*Fill the REQ FIFO of all active channels with one entry */
  for (chan = 1; chan < CHANNELS; chan++) {
    //    printf("size[%d] = %d\n", chan, size[chan]);

    if (active[occ][chan]) {

      /*Write one address to the request FIFO */
      eret = filar_setreq(occ, chan, 1);
      if (eret) {
	printf("Error %d received from setreq for channel %d\n", eret,
	       chan);
	return (1);
      }

    }
  }

  
#ifdef DEBUG3
  printf( "fsize = %d\n\n", fsize );
#endif

  return  fsize;
}



/******************************************************************/
int filar_receive_nonblock_4( int occ, int l2buff, int* filar_err )
/******************************************************************/
{

  long long int cnt = 0, trunc_cnt = 0;
  double *times;
  int *fiforeads;

  double x = 0, x2 = 0;
  double trunc_x = 0, trunc_x2 = 0;
  double avg, rms, first;
  unsigned long long int st, ed;
  double deltat;
  static unsigned int scw = 0xb0f00000, ecw = 0xe0f00000,
    sw1 = 2, nol = 1, nol_save = 10;
  unsigned int fsize, rmode, chan, isready, size[CHANNELS], ffrag,
      complete, ok = 0, bnum;

  static unsigned int expect_size = 126;
  int j, q, output = 0;
  int fragment,  z, err=0, loop=0;

  int ready_count = 0;
  unsigned long long int decision_mask[2];

  unsigned int *ptr, *ptr2, data, data2, data3, eret; // evdata[CHANNELS][0x10000];



  //
  // find the active channels
  //
  data = filar[occ]->ocr;
  active[occ][1] = (data & 0x00000200) ? 0 : 1;
  active[occ][2] = (data & 0x00008000) ? 0 : 1;
  active[occ][3] = (data & 0x00200000) ? 0 : 1;
  active[occ][4] = (data & 0x08000000) ? 0 : 1;
  expect_size = 2;


  //
  // Receive a complete, potentially fragmented packet from all 
  //  active channels
  //
  ffrag = 1;
  complete = fragment = 0;
  for (chan = 1; chan < CHANNELS; chan++)
    size[chan] = 0;
  
  //Wait for a fragment to arrive on all active channels 
  data = filar[occ]->fifostat;
  if( !(data & 0x0000000f) ) // active[occ][1] &&
    return 0;

  loop = 0; 
  while (!complete) 
    { //!complete

#ifdef DEBUG3
      printf( "---> receive loop: \t %d\n", loop ); 
#endif

      if ( fragment )
	{
	  /*Fill the REQ FIFO of all active channels with one entry */
	  for (chan = 1; chan < CHANNELS; chan++) 
	    {
	      if (active[occ][chan]) 
		{
		  /*Write one address to the request FIFO */
		  eret = filar_setreq(occ,chan, 1);
		  if (eret) 
		    {
		      printf("Error %d received from setreq for channel %d\n", eret,
			     chan);
		      return (1);
		    }
		}
	    }
	}
      
 
      

      
      /* Read the ACK FIFOs */
      for (chan = 1; chan < CHANNELS; chan++) {
	if (active[occ][chan]) {
	  if (chan == 1)
	    data = filar[occ]->ack1;
	  if (chan == 2)
	    data = filar[occ]->ack2;
	  if (chan == 3)
	    data = filar[occ]->ack3;
	  if (chan == 4)
	    data = filar[occ]->ack4;
	  data2 = filar[occ]->scw;
	  data3 = filar[occ]->ecw;
	  fsize = data & 0xfffff;
	  
	  
	  if (!(data & 0x20000000)) {
	    complete = 1;
	  } else {
	    fragment = 1;
	    //printf( "Fragment!\n" );
	  }
	  
#ifdef DEBUG3
	  printf( "------------------------------\n");
	  printf( "frag SCW:\t 0x%08x\n", data2 );
	  printf( "frag ECW:\t 0x%08x\n", data3 );
	  printf( "fsize:\t %d\n", fsize );
	  printf( "------------------------------\n");
#endif
	  err = 0;
	  if( data & 0x80000000UL ) 
	    {
	      fprintf( stdout, "s32pci64-filar: Start Control word MISSING\n" );
	      *filar_err = 0x1;
	    }
	  if( data & 0x20000000UL ) 
	    {
	      fprintf( stdout, "End Control word MISSING\n" );
	      *filar_err = 0x2;
	    }
	  if( data2 & 0xF ) 
	    {
	      fprintf( stderr, "Start Control Word Error: 0x%08x  (should be 0x%08x)\n", data2,  SCW );
	      err = 1;
	      *filar_err = 0x4;
	    }
	  if( data3 & 0xF )
	    {
	      fprintf( stderr, "End Control Word Error: 0x%08x  (should be 0x%08x)\n", data3,  ECW );
	      err = 1;
	      *filar_err = 0x8;
	    }

	  

	  ffrag = 0;
	  /*return the buffer and get a pointer to the data */
	  bnum = filar_retbuf(occ,chan, 0);
	  ptr = (unsigned int *) uaddr[occ][chan][bnum];
	  
	  // append frag data to end of user-space buffer
	  if (fsize) {		
	    memcpy( &evdata[occ][l2buff+1][size[chan]], ptr, fsize * 4);
	    size[chan] += fsize;
	  }
	  
#ifdef DEBUG3
	  ptr2 = (unsigned int *) uaddr[occ][chan][bnum];
	  if( fsize )
	    {
	      for( q=0; q<fsize; q++ )
		{
		  printf( "%lx\t", *ptr2 );
		  ptr2++;
		}
	    }
	  printf("\n");
#endif



	  loop++;

	  
	} // if channel active
      } // channel loop
      
#ifdef DEBUG3
      printf( "fragment %d\n", loop );
#endif

    } // while fragment

  /*Fill the REQ FIFO of all active channels with one entry */
  for (chan = 1; chan < CHANNELS; chan++) {

    if (active[occ][chan]) {

      /*Write one address to the request FIFO */
      eret = filar_setreq(occ, chan, 1);
      if (eret) {
	printf("Error %d received from setreq for channel %d\n", eret,
	       chan);
	return (1);
      }

    }
  }

  
#ifdef DEBUG3
  printf( "fsize = %d\n\n", fsize );
#endif
  return  fsize;
}



/********************/
int filar_conf(int occ)
/********************/
{
  static int bswap = 0, wswap = 0, psize = 2, active[MAXFILARS][CHANNELS]; 
                                                   //= { 0, 1, 0, 0, 0 };
  int chan, data;
  char buff[MAXCHAR];

  printf
      ("=============================================================\n");
  data = 0;
  for (chan = 1; chan < CHANNELS; chan++) {
    printf("Enable channel %d (1=yes 0=no) ", chan);
    active[occ][chan] = strtol( fgets( buff, MAXCHAR, stdin ), NULL, 10  ); 
    //active[occ][chan] = getdecd(active[occ][chan]);
    if (!active[occ][chan])
      data += (1 << (6 * chan + 3));
  }
  printf("Select page size:\n");
  printf("0=256 Bytes  1=1 KB  2=2 KB  3=4 KB  4=16 KB\n");
  printf("Your choice ");
  psize = strtol( fgets( buff, MAXCHAR, stdin ), NULL, 10  ); 
  //psize = getdecd(psize);
  printf("Enable word swapping (1=yes 0=no) ");
  wswap = strtol( fgets( buff, MAXCHAR, stdin ), NULL, 10  ); 
  //wswap = getdecd(wswap);
  printf("Enable byte swapping (1=yes 0=no) ");
  bswap = strtol( fgets( buff, MAXCHAR, stdin ), NULL, 10  ); 
  //bswap = getdecd(bswap);

  data += (bswap << 1) + (wswap << 2) + (psize << 3);
  printf("Writing 0x%08x to the OPCTL register\n", data);
  filar[occ]->ocr = data;
  printf
      ("\n=============================================================\n");
}




/*********************************************/
int filar_read_init( int occ, int* filar_err )
/*********************************************/
{

  long long int cnt = 0, trunc_cnt = 0;
  double *times;
  int *fiforeads;

  double x = 0, x2 = 0;
  double trunc_x = 0, trunc_x2 = 0;
  double avg, rms, first;
  unsigned long long int st, ed;
  double deltat;
  static unsigned int scw = 0xb0f00000, ecw = 0xe0f000000,
    sw1 = 2, nol = 1, nol_save = 10;
  unsigned int fsize, rmode, chan, isready, size[CHANNELS], ffrag,
      complete, ok = 0, bnum;

  static unsigned int expect_size = 126;
  int fragment,  z, err=0, loop=0;

  int ready_count = 0;
  unsigned long long int decision_mask[2];

  unsigned int *ptr, *ptr2, data, data2, data3, eret; // evdata[CHANNELS][0x10000];



  data = filar[occ]->ocr;
  active[occ][1] = (data & 0x00000200) ? 0 : 1;
  active[occ][2] = (data & 0x00008000) ? 0 : 1;
  active[occ][3] = (data & 0x00200000) ? 0 : 1;
  active[occ][4] = (data & 0x08000000) ? 0 : 1;
  expect_size = 2;
  
  //
  // Receive a complete, potentially fragmented packet from all 
  //  active channels
  //
  ffrag = 1;
  complete = fragment = 0;
  for (chan = 1; chan < CHANNELS; chan++)
    size[chan] = 0;
  
  /*Fill the REQ FIFO of all active channels with one entry */
  for (chan = 1; chan < CHANNELS; chan++) 
    {
      if (active[occ][chan]) 
	{
	  /*Write 15 addresses to the request FIFO */
	  eret = filar_setreq(occ, chan, 10);
	  if (eret) 
	    {
	      printf("Error %d received from setreq for filar %d channel %d\n", 
		     occ, eret,chan);
	      return (1);
	    }
	}
    }
  
}




/*****************************************/
int filar_read( int occ, int* filar_err )
/*****************************************/
{

  long long int cnt = 0, trunc_cnt = 0;
  double *times;
  int *fiforeads;

  double x = 0, x2 = 0;
  double trunc_x = 0, trunc_x2 = 0;
  double avg, rms, first;
  unsigned long long int st, ed;
  double deltat;
  static unsigned int scw = 0xb0f00000, ecw = 0xe0f000000,
    sw1 = 2, nol = 1, nol_save = 10;
  unsigned int fsize, chan, isready, size[CHANNELS], ffrag,
      complete, ok = 0, bnum;

  static unsigned int expect_size = 126;
  int j, q, output = 0;
  int fragment,  z, err=0, loop=0;

  int ready_count = 0;
  unsigned long long int decision_mask[2];

  unsigned int *ptr, *ptr2, data, data2, data3, eret; // evdata[CHANNELS][0x10000];



  //Wait for a fragment to arrive on all active channels 
  //ready_count=0;
  while (1) {
    
    data = filar[occ]->fifostat;
    isready = 1;
    
    if (active[occ][1] && !(data & 0x0000000f))
      isready = 0;
    if (isready)
      break;
  }
  
  
  for( loop = 0; loop< 15; loop++ )
    {
      
      /* Read the ACK FIFOs */
      for (chan = 1; chan < CHANNELS; chan++) {
	if (active[occ][chan]) {
	  if (chan == 1)
	    data = filar[occ]->ack1;
	  if (chan == 2)
	    data = filar[occ]->ack2;
	  if (chan == 3)
	    data = filar[occ]->ack3;
	  if (chan == 4)
	    data = filar[occ]->ack4;
	  data2 = filar[occ]->scw;
	  data3 = filar[occ]->ecw;
	  fsize = data & 0xfffff;
	  
	  
	  if (!(data & 0x20000000)) {
	    complete = 1;
	  } else {
	    fragment = 1;
	    //printf( "Fragment!\n" );
	  }
	  
#ifdef DEBUG
	  printf( "------------------------------\n");
	  printf( "frag SCW:\t 0x%08x\n", data2 );
	  printf( "frag ECW:\t 0x%08x\n", data3 );
	  printf( "fsize:\t %d\n", fsize );
	  printf( "------------------------------\n");
#endif
	  err = 0;
	  if( data & 0x80000000UL ) 
	    {
	      fprintf( stdout, "s32pci64-filar: Start Control word MISSING\n" );
	      *filar_err = 0x1;
	    }
	  if( data & 0x20000000UL ) 
	    {
	      fprintf( stdout, "End Control word MISSING\n" );
	      *filar_err = 0x2;
	    }
	  if( data2 & 0xF ) 
	    {
	      fprintf( stderr, "Start Control Word Error: 0x%08x  (should be 0x%08x)\n", data2,  SCW );
	      err = 1;
	      *filar_err = 0x4;
	    }
	  if( data3 & 0xF )
	    {
	      fprintf( stderr, "End Control Word Error: 0x%08x  (should be 0x%08x)\n", data3,  ECW );
	      err = 1;
	      *filar_err = 0x8;
	    }
	  
	  

	} // if channel active
      } // channel loop
    } // while fragment

  return  fsize;
}

/********************************************/
int filar_receive_channel( int occ, int* filar_err, int chan )
/********************************************/
{
  static unsigned int scw = 0xb0f00000, ecw = 0xe0f00000;
  unsigned int fsize, rmode, isready, size, ffrag, bnum;
  int j, q, output = 0;
  int fragment, complete, err=0, loop=0;
  unsigned int *ptr, *ptr2, data, data2, data3, eret; 

  // Receive a complete, potentially fragmented packet
  ffrag = 1;
  complete = fragment = 0;
  size = 0;
  
  //Wait for a fragment to arrive
  data = filar[occ]->fifostat;
  if( chan == 1 && !(data & 0x0000000f))
    return 0;
  if( chan == 2 && !(data & 0x00000f00))
    return 0;
  if( chan == 3 && !(data & 0x000f0000))
    return 0;
  if( chan == 4 && !(data & 0x0f000000))
    return 0;

  loop = 0; 
  while (!complete) 
    { //!complete
      if ( fragment )
	{
	  /*Fill the REQ FIFO with one entry */
	  /*Write one address to the request FIFO */
	  eret = filar_setreq(occ,chan, 1);
	  if (eret) 
	    {
	      printf("Error %d received from setreq for channel %d\n", eret,
		     chan);
	      return (1);
	    }
	}
      
      
      /* Read the ACK FIFOs */
      if (chan == 1)
	data = filar[occ]->ack1;
      if (chan == 2)
	data = filar[occ]->ack2;
      if (chan == 3)
	data = filar[occ]->ack3;
      if (chan == 4)
	data = filar[occ]->ack4;
      
      data2 = filar[occ]->scw;
      data3 = filar[occ]->ecw;
      
      fsize = data & 0xfffff;
      
      if (!(data & 0x20000000)) {
	complete = 1;
      } else {
	fragment = 1;
	printf( "Fragment!\n" );
      }
      
      err = 0;
      if( data & 0x80000000UL ) 
	{
	  fprintf( stdout, "Channel %d Start Control word MISSING\n", chan);
	  *filar_err = 0x1;
	}
      if( data & 0x20000000UL ) 
	{
	  fprintf( stdout, "Channel %d End Control word MISSING\n", chan );
	  *filar_err = 0x2;
	}
      
      if( data2 & 0xF ) 
	{
	  fprintf( stderr, "Channel %d Start Control Word Error: 0x%08x  (should be 0x%08x)\n", chan, data2,  SCW );
	  err = 1;
	  *filar_err = 0x4;
	}
      if( data3 & 0xF )
	{
	  fprintf( stderr, "Channel %d End Control Word Error: 0x%08x  (should be 0x%08x)\n", chan, data3,  ECW );
	  err = 1;
	  *filar_err = 0x8;
	}
      
	  

      ffrag = 0;
      /*return the buffer and get a pointer to the data */
      bnum = filar_retbuf(occ,chan, 0);
      ptr = (unsigned int *) uaddr[occ][chan][bnum];
      
      // append frag data to end of user-space buffer
      if (fsize) {		
	memcpy( &evdata[occ][chan][size], ptr, fsize * 4);
	size += fsize;
      }

      loop++;

	  
    } // while fragment

  /*Fill the REQ FIFO  with one entry */
  /*Write one address to the request FIFO */
  //  eret = filar_setreq(occ, chan, 1);
  //  if (eret) {
  //  printf("Error %d received from setreq for channel %d\n", eret, chan);
  //  return (1);
  //  }

  return  fsize;
}

/********************************************/
int filar_receive_channel_ptr( int occ, int* filar_err, int chan, unsigned* userptr)
/********************************************/
{
  unsigned int fsize, bnum;
  unsigned int *ptr, data, data2, data3;

  //Wait for a fragment to arrive
  data = filar[occ]->fifostat;
  if( chan == 1 && !(data & 0x0000000f))
    return 0;
  if( chan == 2 && !(data & 0x00000f00))
    return 0;
  if( chan == 3 && !(data & 0x000f0000))
    return 0;
  if( chan == 4 && !(data & 0x0f000000))
    return 0;

  /* Read the ACK FIFOs */
  if (chan == 1)
    data = filar[occ]->ack1;
  if (chan == 2)
    data = filar[occ]->ack2;
  if (chan == 3)
    data = filar[occ]->ack3;
  if (chan == 4)
    data = filar[occ]->ack4;
  data2 = filar[occ]->scw;
  data3 = filar[occ]->ecw;
  fsize = data & 0xfffff;

  /* Fragment! */
  if ((data & 0x20000000)) {
    *filar_err = 0x16;
  }
  /* Start Control word missing */
  if( data & 0x80000000UL ) 
    {
      *filar_err = 0x1;
    }
  /* End Control word missing */
  if( data & 0x20000000UL ) 
    {
      *filar_err = 0x2;
    }
  /* Start Control word error */
  if( data2 & 0xF ) 
    {
      *filar_err = 0x4;
    }
  /* End Control word error */
  if( data3 & 0xF )
    {
      *filar_err = 0x8;
    }
      
  /*return the buffer and get a pointer to the data */
  bnum = filar_retbuf(occ,chan, 0);
  ptr = (unsigned int *) uaddr[occ][chan][bnum];
  
  // append frag data to end of user-space buffer
  if (fsize) {
    memcpy( userptr, ptr, fsize * 4);
  }

  return  fsize;
}

unsigned *filar_receive_channel_ptr_ncpy( int occ, int* filar_err, int chan, int *fsize)
/********************************************/
{
  unsigned int bnum;
  unsigned int *ptr, data, data2, data3;

  //Wait for a fragment to arrive 
/*   data = filar[occ]->fifostat; */

/*   if( chan == 1 && !(data & 0x0000000f)) */
/*     return 0; */
/*   if( chan == 2 && !(data & 0x00000f00)) */
/*     return 0; */
/*   if( chan == 3 && !(data & 0x000f0000)) */
/*     return 0; */
/*   if( chan == 4 && !(data & 0x0f000000)) */
/*     return 0; */



  /* Read the ACK FIFOs */
  if (chan == 1)
    data = filar[occ]->ack1;
  if (chan == 2)
    data = filar[occ]->ack2;
  if (chan == 3)
    data = filar[occ]->ack3;
  if (chan == 4)
    data = filar[occ]->ack4;

  
/*   data2 = filar[occ]->scw; */
/*   data3 = filar[occ]->ecw; */
  
  *fsize = data & 0xfffff;

  /* Fragment! */
  if ((data & 0x20000000)) {
    *filar_err = 0x16;
  }
  /* Start Control word missing */
  if( data & 0x80000000UL ) 
    {
      *filar_err = 0x1;
    }
  /* End Control word missing */
  if( data & 0x20000000UL ) 
    {
      *filar_err = 0x2;
    }
  /* Start Control word error */
  
/*   if( data2 & 0xF )  */
/*     { */
/*       *filar_err = 0x4; */
/*     } */
  
  /* End Control word error */
  
/*   if( data3 & 0xF ) */
/*     { */
/*       *filar_err = 0x8; */
/*     } */
  
  /*return the buffer and get a pointer to the data */
  bnum = filar_retbuf(occ,chan, 0);
  ptr = (unsigned int *) uaddr[occ][chan][bnum];

  if( !ptr ) 
    {
      *filar_err = 0x4;
    }


/*   printf("fsize = %d ",*fsize);   */
/*   printf("ptr = %x\n",ptr); */
  return ptr;

  // append frag data to end of user-space buffer
/*   if (fsize) { */
/*     memcpy( userptr, ptr, fsize * 4); */
/*   } */

/*   return  fsize; */
}



unsigned int filar_receive_all( int occ)
/********************************************/
{
  unsigned int data;
  unsigned int receive = 0;

  //Wait for a fragment to arrive 
  data = filar[occ]->fifostat;

  if( data & 0x0000000f )
    receive |= 0x1;
  if( data & 0x00000f00 )
    receive |= 0x2;
  if( data & 0x000f0000 )
    receive |= 0x4;
  if( data & 0x0f000000 )
    receive |= 0x8;

  return receive;

}

/************************/
void filar_reset(int occ) 
/************************/
{ 

  fprintf( stderr, "Filar::filar_reset(%d)\n", occ  );  
  int c=0;
  
  fprintf( stderr, "Filar::filar_cardreset(%d)\n", occ  );  
  filar_cardreset(occ);
  fprintf( stderr, "Filar::filar_linkreset(%d)\n", occ  );  
  filar_linkreset(occ);
  
  eret = filar_setreq(occ, 1, 4);
  if (eret) {
    fprintf(stderr,"Filar::Error %d received from setreq for channel %d\n", eret,
	    1);
    return;
  }

}


/************************/
void filar_reset_hola(int occ,  int channel_mask) 
/************************/
{ 
  int chan=0;

  fprintf( stderr, "Filar::filar_reset(%d)\n", occ  );  
  int c=0;
  
  fprintf( stderr, "Filar::filar_cardreset(%d)\n", occ  );  
  filar_cardreset(occ);
  fprintf( stderr, "Filar::filar_linkreset(%d)\n", occ  );  
  fprintf( stderr, "channel mask = 0x%08x\n", channel_mask  );  
  fprintf( stderr, "filar_linkreset(%d)\n", occ  );  
  for (chan = 0; chan < CHANNELS; chan++) {
   if( channel_mask & (1<<chan) )
    filar_linkreset_hola(occ, chan);
  }
  //  filar_linkreset(occ);
  
  eret = filar_setreq(occ, 1, 4);
  if (eret) {
    fprintf(stderr,"Filar::Error %d received from setreq for channel %d\n", eret,
	    1);
    return;
  }

}
