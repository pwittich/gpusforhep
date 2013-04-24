
ifeq ($(ARCH), i686) # L2GPU
	TARGETS = test_svt.exe main
else
	TARGETS = timing.exe
endif

all: $(TARGETS)

timing.exe: timing.o
	$(CC) $(CCFLAGS) $< -o ../../bin/timing.exe -L../../lib -ll2slink -lrt

main: trackfit.o main.o
	gcc $(CCFLAGS) $^ -o ../../bin/main $(LDFLAGS) $(LIBS)

test_svt.exe: test_svt.o kernel_SVT.o svtsim_functions.o
	gcc $(CCFLAGS) $^ -o ../../bin/test_svt.exe $(LDFLAGS) $(LIBS)

trackfit.o: trackfit.cu trackfit_kernel.cu
	$(NVCC) $(NVFLAGS) -I/opt/cuda/include -c $<

kernel_SVT.o: kernel_SVT.cu svtsim_functions.c svtsim_functions.h functionkernel.h
	$(NVCC) $(NVFLAGS) -I/opt/cuda/include -c $<

main.o: main.c main.h trackfit_host.c
	gcc $(CCFLAGS) $(IFLAGS) -c $<

test_svt.o: test_svt.c svtsim_functions.c svtsim_functions.h functionkernel.h
	gcc $(CCFLAGS) $(IFLAGS) -w -lm $^ -c $<

svtsim_functions.o: svtsim_functions.c svtsim_functions.h functionkernel.h
	gcc -w -c $<

timing.o: timing.cc
	$(CC) $(CCFLAGS) $(IFLAGS) -c $<

clean:
	rm -f main *.o *.linkinfo

run:
	rm -f main *.o *.linkinfo
	make all
	./main
