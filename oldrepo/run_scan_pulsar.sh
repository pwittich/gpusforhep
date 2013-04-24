#!/bin/sh

N_LOOPS=10000;
N_WORDS=512;
#N_THREADS=(32 64 160 320 640 1600 3200 6400 9600 12800 16000)
N_THREADS=(32 64 128 256 512 1024 2048 4096 8192 16384)

for i in ${N_THREADS[@]}
do
    echo $i
    sudo bin/timing.exe $N_LOOPS $N_WORDS output/$i.txt >& output/log_$i.txt
    sleep 5 # make sure gpu is ready
done

