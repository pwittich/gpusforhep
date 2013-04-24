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

#define TICKS_PER_USEC 2789.010 // from /proc/cpuinfo 
#define TICKS_PER_100NSEC 278.9010 // from /proc/cpuinfo 



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


//
//  useful if we choose to add error strings to theerror objects
//  not yet used

#include <sstream>
#include <string>

template <class T>
std::string to_string(T t, std::ios_base & (*f)(std::ios_base&))
{
  std::stringstream ss;
  ss << f << t;
  return ss.str();
};

//
// 04/04/2008 pmf: get local TED control ip address from environment
//
inline char *TedIPlocal() {
  char c_ipaddr[256] = "192.168.0.1";
  const char* c_ipaddr_test = getenv("TEDCONTROL_IPLOCAL");
  if( c_ipaddr_test==NULL ) {
    fprintf( stderr, "TedIPlocal(): Failed getting local TED control IP address from $TEDCONTROL_IPLOCAL. Using default %s\n", c_ipaddr);
    }
  else if( strcmp( c_ipaddr_test,"192.168.0.20")==0 ||
           strcmp( c_ipaddr_test,"192.168.0.21")==0 ||
           strcmp( c_ipaddr_test,"192.168.0.1" )==0 ) {
    strcpy(c_ipaddr, c_ipaddr_test);
    fprintf( stderr, "TedIPlocal(): Fetching local TED control IP %s from $TEDCONTROL_IPLOCAL\n", c_ipaddr);
    }
  else {
    fprintf( stderr, "TedIPLocal(): $TEDCONTROL_IPLOCAL has invalid value %s - ignored. Using default %s\n", c_ipaddr_test, c_ipaddr);
    }
  fflush( stderr );
return c_ipaddr;
}

//
// 04/04/2008 pmf: lock file handling with TED control node info for PROCMON
//
inline void dcsn_lock() {
  if( FILE *flock=fopen( "/tmp/.dcsn_lock", "r" ) )  {
    fprintf( stderr, "dcsn_lock(): Warning, lock file /tmp/.dcsn_lock found - might indicate that another l2node.exe process is still running.\n" );
    fclose( flock );
    }
  if( FILE *flock=fopen( "/tmp/.dcsn_lock", "w" ) )  {
    fprintf( flock, "%s", TedIPlocal() ) ;
    fprintf( stderr, "dcsn_lock(): New lock file /tmp/.dcsn_lock successfully established.\n" );
    fclose( flock );
    }
  else {
    fprintf( stderr, "dcsn_lock(): Failed creating new lock file /tmp/.dcsn_lock. This might confuse PROCMON. Please check manually.\n" );
    }
  fflush( stderr);
}
inline void dcsn_unlock() {
  if( FILE *flock=fopen( "/tmp/.dcsn_lock", "r" ) )  {
    if( remove( "/tmp/.dcsn_lock" ) == 0 )
   	 fprintf( stderr, "dcsn_unlock(): Lock file /tmp/.dcsn_lock removed.\n" );
    else
         fprintf( stderr, "dcsn_unlock(): Failed removing lock file /tmp/.dcsn_lock.\n" );
    fclose( flock );
    }
  else {
    fprintf( stderr, "dcsn_unlock(): No lock file /tmp/.dcsn_lock found - might already have been removed in previous attempts.\n" );
    }
  fflush( stderr);
}
#endif
