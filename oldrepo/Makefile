
ARCH := $(shell uname -m)
ifeq ($(ARCH), i686)
	SLINK_DIR = slink_gpu
else
	SLINK_DIR = slink_pulsar
endif

# package definitions
ROOT_DIR = $(PWD)
OBJ_DIR = $(ROOT_DIR)/obj
DIRS = $(ROOT_DIR)/$(SLINK_DIR) $(ROOT_DIR)/slinktest
DRIVERS = cmem_rcc.o io_rcc.o

##########################################################################################

# base make and compiler options (  appended to/overridden in submakes )

MAKE   = gmake  -j --no-print-directory
SHELL    = /bin/bash

CC     = g++ -D __KERNEL_STRICT_NAMES
CCFLAGS = -ggdb -O2 -w -I/usr/src/linux/include -fPIC -fPIE -ffast-math -funroll-loops -fvisibility=protected
#CCFLAGS += -g -pg # for gprof
LIBS   = -ll2slink -lpthread -ldl

export CC CCFLAGS LIBS BIN OBJ ROOT_DIR EXE DRIVERS MAKE ARCH SLINK_DIR

##########################################################################################
# rules
% : 
	@if [ ! -d obj ]; then mkdir obj; fi;
	@if [ ! -d bin ]; then mkdir bin; fi;
	@if [ ! -d lib ]; then mkdir lib; fi;
	@if [ '$@' == 'clean' ]; then rm -f lib/lib*; rm -f bin/*; rm -f drivers/*;  fi	
	@for i in $(DIRS); \
	do cd $$i && $(MAKE) $@; \
	done; 

