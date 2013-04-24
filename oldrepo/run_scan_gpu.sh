#!/bin/sh

N_LOOPS=10000;
N_WORDS=512;
#N_THREADS=(32 64 160 320 640 1600 3200 6400 9600 12800 16000)
N_THREADS=(32 64 128 256 512 1024 2048 4096 8192 16384)
#METHOD="ALG"
#METHOD="COPY"
#METHOD="WEAVE"
#METHOD="CPU"
METHOD="SP"
RUN=0

mkdir -p output

for i in ${N_THREADS[@]}
do
    echo $i
    sudo ./loaddrivers
    sudo bin/main $N_LOOPS $N_WORDS $i $METHOD $RUN | tee output/log_${N_WORDS}W_${i}T_${METHOD}_${RUN}.txt
done
