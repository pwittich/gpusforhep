#!/bin/bash

# for setting IRQ (Interrupt ReQuest) affinities. default was ff
# note that 0 and 2 are special and the kernel will not
#   allow them to be changed

maskGPU="fe" # allow IRQ on all but processor 1
maskPULSAR="00000002"

if [ `uname -m` == "i686" ]; then
    mask=$maskGPU
else
    mask=$maskPULSAR
fi

path="/proc/irq/*/smp_affinity"

echo "-- Current SMP affinities: "
sudo grep -H . $path

if [ ! $USER == "root" ]; then
    echo "Warning: must run as root (not sudo) to write"
    exit
fi

echo "-- Setting new SMP affinities..."

for i in `ls -1 $path`; do
    echo -n $mask > $i
done

echo "-- New SMP affinities: "
sudo grep -H . $path

