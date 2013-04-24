#!/bin/sh

N_LOOPS=10000;
#N_WORDS=500;
#N_THREADS=3200;
N_WORDS=512
N_THREADS=16384
N_TIMES=10;

for i in `seq 1 $N_TIMES`
do
    echo $i
    sudo bin/timing.exe $N_LOOPS $N_WORDS output/${N_THREADS}_$i.txt >& output/log_${N_THREADS}_$i.txt
    sleep 5 # make sure gpu is ready
done

