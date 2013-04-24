// gcc -w -lm svtsim_functions.c test_svt.c -o test_svt.exe

#include <sched.h>
#include <linux/types.h>
#include <time.h>
#include "NodeUtils_basic.hh"

#include "svtsim_functions.h"
#include "functionkernel.h"
#include "semaphore.c"


#define GPU 1
//#define TIMING
#define PRIORITY

#define n_iter 100 // number of iterations for timing measurements

int main()
{
    // lock control so no one else can run at the same time and crash the machine
    key_t key = (key_t) 0xdeadface;
    int semid;

    if ((semid = initsem(key, 1)) == -1) {
        perror("initsem");
        exit(1);
    }
    printf("Trying to gain control...\n");
    lock(semid);
    // locked

#ifdef PRIORITY
    // set scheduling priority & CPU affinity
    struct sched_param p;
    p.sched_priority = 99;
    if (sched_setscheduler(0, SCHED_FIFO, &p)) {
	perror("setscheduler");
	return -1;
    }
    if (sched_getparam(0, &p) == 0)
	printf("Running with scheduling priority = %d\n", p.sched_priority);
    
    unsigned long mask;
    if (sched_getaffinity(0, sizeof(mask), (cpu_set_t*)&mask) < 0) {
	perror("sched_getaffinity");
    }
    printf("my affinity mask is: %08lx\n", mask);
    
    mask = 1; // processor 1 only
    if (sched_setaffinity(0, sizeof(mask), (cpu_set_t*)&mask) < 0) {
	perror("sched_setaffinity");
	return -1;
    }

    if (sched_getaffinity(0, sizeof(mask), (cpu_set_t*)&mask) < 0) {
	perror("sched_getaffinity");
    }
    printf("my affinity mask is: %08lx\n", mask);
#endif
    
    //const char *finPath  = "slinktest/data/hbout_w6_100evts";
    //const char *finPath  = "slinktest/data/hbout_100evt_32r_2hl";
    //const char *finPath  = "slinktest/data/hbout_2evt_4r_2hl";
    const char *finPath  = "slinktest/data/hbout_10evt_64r_2hl";
    //const char *finPath  = "slinktest/data/hbout_100evt_64r_2hl";
    //const char *finPath  = "slinktest/data/hbout_1000evt_64r_2hl";
    const char *foutPath = "slinktest/data/test_svt_out";
    
    FILE *fin  = fopen(finPath, "r");
    FILE *fout = fopen(foutPath, "w");

    // get number of words (lines)
    int ch;
    unsigned n_words = 0;
    while (EOF != (ch=getc(fin)))
	if (ch=='\n')
	    ++n_words;
    rewind(fin);
    printf("read n_words = %u from %s\n", n_words, finPath);

    
    // read into array
    unsigned *data = malloc(n_words * sizeof(unsigned));
    char line[80];
    unsigned word;
    int i=0;
    while (fgets(line, 80, fin) != NULL) {
	sscanf(line, "%6x", &word);
	//printf("%.6x\n", word);
	data[i++] = word;
    }

    //printf("data[0] = %.6x\n", data[0]);
    //printf("data[1] = %.6x\n", data[1]);

    tf_arrays_t tf;

    printf("sizeof(tf_arrays) = %u\n", sizeof(struct tf_arrays));
    printf("sizeof(tf) = %u\n", sizeof(tf));

    if (GPU) {
	GPU_Init(n_words);
	
	gf_init(&tf);
	svtsim_fconread(tf);

	//printf("from main: %u\n", tf->gf_emsk);
	//printf("%d\n", tf);
	//launchTestKernel_tf(tf, data, n_words);

	gf_fep_gpu(tf, n_words, data);
	
#ifdef TIMING
	int i;
	__u32 start, finish;
	rdtscl(start);
	for (i = 0; i < n_iter; i++)
	    gf_fep_gpu(tf, n_words, data);
	rdtscl(finish);
        float time_us = tstamp_to_us(start, finish);
	printf("Time for gf_fep_gpu: %f us/run\n", time_us / n_iter);
#endif

	gf_fit(tf);
	gf_comparator(tf);

	GPU_Destroy();
    } else {
	gf_init(&tf);
	svtsim_fconread(tf);
	gf_fep(tf, n_words, data);

#ifdef TIMING
	int i;
	__u32 start, finish;
	rdtscl(start);
	for (i = 0; i < n_iter; i++)
	    gf_fep(tf, n_words, data);
	rdtscl(finish);
        float time_us = tstamp_to_us(start, finish);
	printf("Time for gf_fep: %f us/run\n", time_us / n_iter);
#endif

	
	gf_fit(tf);
	gf_comparator(tf);
    }	

    // dump the output "cable"
    svtsim_cable_t *out = tf->out;
    for (i=0; i<out->ndata; i++) {
	fprintf(fout, "%.6x\n", out->data[i]);
    }

    printf("output written to %s\n", foutPath);

    fclose(fin);
    fclose(fout);
    
    return 0;
}
