
/***************************************
 * Defines for filar/solar that are    *
 * specific to our (CDF-L2) use        *
 ***************************************/

/*** filar ***/
#define MAXFILARS    2
#define CHANNELS     2		 /* we will not use channel[0] */
#ifndef F_BUFSIZE
#define F_BUFSIZE    (4*576)	 /* bytes for the event data */
#endif


/*** solar ***/
#define MAXSOLARS    1
#define MAXBUF       1
#define MAXREQ       15
#ifndef S_BUFSIZE
#define S_BUFSIZE    ( 4*4 )      /* bytes for the decision */
#endif


/*** common ***/
#define SCW          0xB0F00000
#define ECW          0xE0F00000
#define PREFILL      0xdeadbeef


