#ifndef L2UTILS
#define L2UTILS

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 05/05/08 pmf: added useful functions


//
// timing stuff
//

// these are from the old pc  
//#define TICKS_PER_USEC 2605.933 // from /proc/cpuinfo 
//#define TICKS_PER_100NSEC 260.5933 // from /proc/cpuinfo 

//for new PC
//#define TICKS_PER_USEC 2789.010 // from /proc/cpuinfo 
//#define TICKS_PER_100NSEC 278.9010 // from /proc/cpuinfo 

//for table-top PC
#define TICKS_PER_USEC 3073.780 // from /proc/cpuinfo 
#define TICKS_PER_100NSEC 307.3780 // from /proc/cpuinfo 


#define rdtscl(low) \
     __asm__ __volatile__ ("rdtsc" : "=a" (low) : : "edx")



inline float tstamp_to_us(unsigned tstart, unsigned tend )
{
  if (tend > tstart)
    return (((float)(tend-tstart))/TICKS_PER_USEC);
  else
    return (((float)(0xffffffff-(tstart-tend)))/TICKS_PER_USEC);
}

/*
inline int tstamp_to_us(unsigned tstart, unsigned tend )
{
  if (tend > tstart)
    return ((tend-tstart)/TICKS_PER_USEC);
  else
    return ((0xffffffff-(tstart-tend))/TICKS_PER_USEC);
}
*/

inline int tstamp_to_ms(unsigned tstart, unsigned tend )
{
  if (tend > tstart)
    return ((tend-tstart)/(TICKS_PER_USEC*1000));
  else
    return ((0xffffffff-(tstart-tend))/(TICKS_PER_USEC*1000));
}

inline double tstamp_to_fp_ms(unsigned tstart, unsigned tend )
{
  if (tend > tstart)
    return ((tend-tstart)/TICKS_PER_USEC*1000);
  else
    return ((0xffffffff-(tstart-tend))/(TICKS_PER_USEC*1000));
}

inline unsigned int tstamp_to_100ns(unsigned tstart, unsigned tend )
{
  if (tend > tstart)
    return ((tend-tstart)/TICKS_PER_100NSEC);
  else
    return ((0xffffffff-(tstart-tend))/TICKS_PER_100NSEC);
}


#endif
