#!/bin/sh

N_LOOPS=10000;
#N_WORDS=500;
#N_THREADS=3200;
N_WORDS=512
N_THREADS=16384
N_TIMES=10;
#METHOD="ALG";
METHOD="COPY";
#METHOD="SP";

for i in `seq 1 $N_TIMES`
do
    echo $i
    sudo ./loaddrivers
    sudo bin/main $N_LOOPS $N_WORDS $N_THREADS $METHOD $i | \
	tee output/log_${N_WORDS}W_${N_THREADS}T_${METHOD}_$i.txt
done
