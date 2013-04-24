##########################################################################################
#
# l2node Makefile
#
# pass targets to a submake in src
#
##########################################################################################

# variables

SDKPATH := /opt/cuda/sdk
CUDAPATH := /opt/cuda
NVCC := $(CUDAPATH)/bin/nvcc

SRC=   solarsend.cc filarreceive.cc TrackFit.cc 
OBJ=  $(subst .cc,.o,$(SRC) )

IFLAGS = -I /usr/include \
	-I ../../$(SLINK_DIR)/s32pci64-solar/s32pci64-solar \
	-I ../../$(SLINK_DIR)/s32pci64-filar/s32pci64-filar \
	-I ../include \
	-I$(CUDAPATH)/include \
        -I$(SDKPATH)/common/inc

#LDFLAGS = -L$(ROOT_DIR)/lib -L$(CUDAPATH)/lib -L$(SDKPATH)/C/lib -L$(SDKPATH)/CUDALibraries/common/lib/linux -lcutil -lcudpp -lcuda -lcudart
LDFLAGS = -L$(ROOT_DIR)/lib -L$(CUDAPATH)/lib -L$(SDKPATH)/C/lib -L$(SDKPATH)/CUDALibraries/common/lib/linux -lcuda -lcudart
CCFLAGS += -DCOMM  $(IFLAGS) 
NVFLAGS = -O3 -arch=sm_13 # we use devices with compute capability 1.3 or higher. allows compiler to make optimizations
#NVFLAGS += --profile

export CC MAKE CCFLAGS IFLAGS LIBS LDFLAGS ROOT_DIR LOCAL_ROOT_DIR EXE PKGDEP SRC OBJ NUMADIR NUMALIBS DLL SDKPATH CUDAPATH NVCC NVFLAGS

##########################################################################################

# rules

% :
	echo "DLL = $(DLL)"
	if [ ! -e obj ]; then mkdir obj; fi;
	if [ ! -e bin ]; then mkdir bin; fi;
	cd src && $(MAKE) $@
