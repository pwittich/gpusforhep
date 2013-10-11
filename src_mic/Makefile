
INTELBASE=/software/intel_parallel_studio_xe_2013/composer_xe_2013.3.163/bin/intel64

ICC=$(INTELBASE)/icc
ICPC=$(INTELBASE)/icpc

INCTHRUST=/home/gianelle/thrust-1.7

ICFLAGS= -openmp -O3 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DNDEBUG


%.o: %.c
	$(ICC) $(ICFLAGS) -I. -c $< -o $@
%.o: %.cc
	$(ICPC) $(ICFLAGS) -I$(INCTHRUST) -I. -c $< -o $@

all: svt_mic

svt_mic: svtsim_functions.o svt_mic.o
	$(ICPC) $(ICFLAGS) $^ -o $@ -rdynamic


clean:
	$(RM) *.o *~ svt_mic

