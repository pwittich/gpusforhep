
INTELBASE=/opt/intel/bin

ICC=$(INTELBASE)/icc
ICPC=$(INTELBASE)/icpc

INCTHRUST=/usr/local/cuda-5.0/include

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

