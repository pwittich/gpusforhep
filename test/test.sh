#!/bin/bash

# Set base path
BASE=/home/gianelle/svt/svn/gpusforhep/trunk

SRC=${BASE}/src
TEST=${BASE}/test

CPU=""
# If you want to run tests on cpu uncomment line below
#CPU="-s cpu"

# number of loops per run
LOOPS=1000

# which event's files we want to tests
EVT="1 10 50 100 200 500 1000 1500 2000 2500 3000"

for e in $EVT ; do

  echo " --> Test on $e events"
  echo " -> Compiling test..."

  cd ${SRC}
  cat svtsim_functions.h | sed -e "s/#define NEVTS .*/#define NEVTS ${e}/" > tmp
  mv tmp svtsim_functions.h
  make clean
  make

  echo " -> Run tests..."
  cd ${TEST}
  ${SRC}/svt_gpu ${CPU} -t -i data/test_${e}e_64r_32c.in -l $LOOPS

  echo " -> Check output..."
  sort gfout.txt > my.out
  sort data/test_${e}e_64r_32c.out > good.out
  diff my.out good.out > /dev/null
  if [ $? -eq 1 ] ; then
    echo "ERROR. Run with $e events failed!"
    rm my.out good.out
    exit 1
  else
    rm my.out good.out
  fi
  
done 

echo " --> All done!"

