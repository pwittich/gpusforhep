CC=gcc
#CCFLAGS=-std=gnu99 -g -DDEBUG_SVT
CCFLAGS=-std=gnu99 -O3

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

all: svt

svt: svt.o  svtsim_functions.o 
	$(CC) $^ -o $@ -lm

s2: s2.o
	$(CC) $^ -o $@ 


clean:
	$(RM) *.o *~ svt_gpu

depend:
	makedepend -Y $(INCFLAG) *.cu *.cc *.h *.c

# DO NOT DELETE

svt_utils.o: svtsim_functions.h
svt.o: svt_utils.h svtsim_functions.h
svtsim_functions.o: svtsim_functions.h
