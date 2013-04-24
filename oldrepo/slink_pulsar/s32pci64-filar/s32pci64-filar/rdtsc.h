/* get number of clock cycles since last boot */

/* expects long long as argument */
#define rdtsc(x)  __asm__ volatile ("rdtsc": "=A" (x));

