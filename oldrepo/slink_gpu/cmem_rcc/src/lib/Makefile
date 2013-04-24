#################################################################

OBJ     = cmem_rcc_lib.o

CWARN   += -Wall -Wstrict-prototypes -Wpointer-arith -Wcast-qual \
	      -Wcast-align -Wconversion 
CFLAGS  += $(CWARN) -g -funsigned-char 

# due to differences between L2GPU and L2PULSAR
ifeq ($(ARCH), i686)
	CFLAGS += -DL2GPU
endif

#################################################################

all: lib

lib: $(OBJ)
	ar rv $(LOCAL_LIB) $< 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

dep:
	@if [ ! -e .dep ]; then touch .dep;  fi
	makedepend -Y -f .dep $(IFLAGS) *.c >& /dev/null

clean:
	- rm *.o *.a
	- rm -f .dep
	- ar dv $(LOCAL_LIB) $(OBJ)

#################################################################

-include .dep
