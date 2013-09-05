CPP=g++
RM=rm -f
#CPPFLAGS=-Wall
CPPFLAGS=-w -fpermissive

IFLAGS=	-I./
LDLIBS=-lOpenCL

%.o: %.cpp
	$(CPP) $(CPPFLAGS) $(IFLAGS) -c $< -o $@

all: svt_gpu_opencl

svt_gpu_opencl: svt_gpu_opencl.o gf_unpack.o gf_fep.o gf_fit.o svtsim_functions.o helperFuncs.o clHelperFuncs.o
	$(CPP) $^ -o $@ $(LDLIBS)

clean:
	$(RM) *.o *~

# DO NOT DELETE

