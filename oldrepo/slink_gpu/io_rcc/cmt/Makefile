export kernel_version=$(shell uname -r)

all::
	@if [ ! -f ../$(CMTCONFIG)/io_rcc-${kernel_version}.c ]; then \
	  ln -s ../src/driver/io_rcc_driver.c ../$(CMTCONFIG)/io_rcc-${kernel_version}.c ; \
	fi

include $(CMTROOT)/src/Makefile.header

include $(CMTROOT)/src/constituents.make
