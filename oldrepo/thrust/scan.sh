#! /bin/sh -x
# run a scan over various input data sizes

NWS=" 16 17 18 19 20 21 22 "

for n in $NWS
do
    ./markpack -s $n 1> out_${n}.dat
done
