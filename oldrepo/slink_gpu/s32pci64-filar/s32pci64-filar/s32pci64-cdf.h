
/***************************************
 * Defines for filar/solar that are    *
 * specific to our (CDF-L2) use        *
 ***************************************/

/*** filar ***/
#define MAXFILARS    3
#define CHANNELS     5	 /* we will not use channel[0] */
#ifndef F_BUFSIZE
#define F_BUFSIZE    2048 /* bytes for the event data */
#endif
#define MAXBUF       32
#define MAXREQ       32


/*** solar ***/
#define MAXSOLARS    1
#ifndef S_BUFSIZE
#define S_BUFSIZE    2048
#endif


/*** common ***/
#define SCW          0xB0F00000
#define ECW          0xE0F00000
#define PREFILL      0xdeadbeef


