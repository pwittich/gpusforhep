//
// system headers
//

#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sched.h>
#include <vector>
#include <linux/types.h>

#include "NodeUtils.hh"

#include "semaphore.c"

//
// s32pci64 headers
//
extern "C" {
#include "s32pci64-filar.h"
#include "s32pci64-solar.h"
}

#define FILAR_NUMBER_0 0
#define FILAR_CHANNEL_0 0x1
#define FILAR_CHANNEL_MASK 0x1

//#define N_LOOPS  1000
//#define N_WORDS 10

#define rdtscl(low) \
    __asm__ __volatile__ ("rdtsc" : "=a" (low) : : "edx")

using namespace std;
int main(int argc, char *argv[])
{

    //for storing timing
    char *file_name;
    ofstream file_timeout;
    if (argc < 3 || argc > 4) {
        cout << "Usage: timing.exe N_LOOPS N_WORDS OUTPUT_FILE" << endl;
        return 0;
    }

    // lock control so no one else can run at the same time and crash the machine
    key_t key = (key_t) 0xdeadface;
    int semid;

    if ((semid = initsem(key, 1)) == -1) {
        perror("initsem");
        exit(1);
    }
    printf("Trying to gain control...\n");
    lock(semid);

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
    


    const int N_LOOPS = atoi(argv[1]);
    const int N_WORDS = atoi(argv[2]);
    if (argc == 4) {
        file_name = argv[3];
    } else {
        time_t rawtime; struct tm *timeinfo;
        time(&rawtime); timeinfo = localtime(&rawtime);
        strftime(file_name, 80, "timing_LOGS/timing_%b%d_%H%M_%S.log", timeinfo);
        //file_timeout.open("time_log.txt");
        puts(file_name);
    }
    file_timeout.open(file_name);


    __u32 time_recv;
    __u32 time_send;

    struct timespec ts_s, ts_e;

    unsigned int *dataptr_solar;
    // return packets?
    unsigned int buff[12];
    //
    // set up the s32pci64 cards
    //
    cerr << "before solar_setup(0)" << endl;
    solar_setup(0);
    dataptr_solar = solar_getbuffer(0);

    unsigned *dataptr_filar;
    unsigned data[N_WORDS];
    unsigned data_send[N_WORDS];

    int kf;
    srand(100);
    //for(kf=0;kf<500;kf++)
    // data_send[kf]=0xcdf1000 + kf;
    //data_send[1]= ( rand() % 0xff);

    dataptr_filar = &(data[0]);


    cerr << "before filar_setup"  << endl;
    // Yo, if you want this to work comment out the while loop
    // in linkreset!!!  It checks for recovery off all 4 channels
    filar_setup_hola(FILAR_NUMBER_0, FILAR_CHANNEL_MASK);
    cerr << "after filar_setup"  << endl;


    int reqfifobufcount = 0;
    int reqfifobufused = 1;


    // ARM REQ FIFOs before going to event loop
    //  filar_setreq(FILAR_NUMBER_0, FILAR_CHANNEL_0, reqfifobufused);

    filar_setreq(FILAR_NUMBER_0, FILAR_CHANNEL_0, 4);


    //
    //  event loop
    // ---------------------------
    int nevent = 1;
    int filar_err;
    int count = 0, dumpcount = 0;
    float times_array[N_LOOPS];

    while (count < N_LOOPS) {
        // set things up first
        // first for the filar
        filar_err = 0;
        int fsize = 0;
        // Write REQ FIFO for Cluster channel
        if (reqfifobufcount == reqfifobufused - 1) {
            filar_setreq(FILAR_NUMBER_0, FILAR_CHANNEL_0, reqfifobufused);
        }

        //then for the solar...
        for (kf = 0; kf < N_WORDS; kf++) {
            data_send[kf] = (rand() % 0xffffffff);
            //for(kf=0;kf<500;kf++){
            //data_send[kf]++;
            //cout << "Taking word " << 1 << ": "<< hex << data_send[0] << endl;
            //data_send[kf]+=rand();
            //cout << " ... now " << hex << data_send[kf] << endl;
        }

        //usleep(10000); // why? -stp
	usleep(1000); // why do we freeze without this? -stp

        rdtscl(time_send);
	//clock_gettime(CLOCK_MONOTONIC, &ts_s);
	
        //send something on solar...
        solar_send_ptr(N_WORDS, (unsigned *)&data_send, 0);

        //now we wait
        while (!fsize)
            fsize = filar_receive_channel_ptr(FILAR_NUMBER_0, &filar_err, FILAR_CHANNEL_0, (unsigned *)(&data));

	//clock_gettime(CLOCK_MONOTONIC, &ts_e);
        rdtscl(time_recv);

        //for( int j=0; j<fsize; j++ )
        //    cout << "word " << dec << j << " : 0x" << hex <<  *(dataptr_filar+j) << endl;

        count++;
        //if (count%10==0)
        cout << "Finished " << count << " patterns. Got fsize=" << dec << fsize << endl;

        float time_us = tstamp_to_us(time_send, time_recv);
	//float time_us_timer = delta_time_us(ts_s, ts_e);
        //file_timeout << dec << time_recv - time_send << "\t" << time_us << "\t" << time_us_timer << endl;
	file_timeout << dec << time_recv - time_send << "\t" << time_us << endl;

        times_array[count - 1] = time_us;

        cout << flush;
        reqfifobufcount++;
        if (reqfifobufcount == reqfifobufused) {
            reqfifobufcount = 0;
        }

    } // N_LOOPS

    file_timeout.close();

    float mean = 0;
    for (int k = 0; k < N_LOOPS; k++) {
        mean += times_array[k] / (float)N_LOOPS;
    }
    float stdev = 0;
    for (int k = 0; k < N_LOOPS; k++) {
        stdev += (times_array[k] - mean) * (times_array[k] - mean) / (float)(N_LOOPS);
    }
    stdev = sqrt(stdev);
    cout << "Mean latency for this run = " << dec << mean << endl;
    cout << "St.Dev. of latency for this run = " << dec << stdev << endl;


    printf("Unlocking control...\n");
    unlock(semid);

    return 1;

}
