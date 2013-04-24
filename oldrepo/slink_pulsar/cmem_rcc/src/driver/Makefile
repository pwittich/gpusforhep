#************************************************************
# Makefile for linux IO_RCC driver
#
# 07/01/08 last mod: pmf
#************************************************************

KERNELDIR = /usr/src/linux
IFLAGS	:= -I../.. -I$(KERNELDIR)/include
KCFLAGS	:= -DKBUILD_MODNAME="io_rcc_driver" 
obj-m 	:= cmem_rcc_drv.o

.DEFAULT: all

clean: 
	- rm -f *.o *.ko *.mod.c
