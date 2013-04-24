#************************************************************
# Makefile for linux IO_RCC driver
#
#************************************************************

KERNELDIR = /usr/src/linux
IFLAGS  := -I../.. -I$(KERNELDIR)/include
KCFLAGS := -DKBUILD_MODNAME="io_rcc_driver"
obj-m   := io_rcc_driver.o

.DEFAULT: all
clean: 
	- rm -f *.o *.ko *.mod.c
