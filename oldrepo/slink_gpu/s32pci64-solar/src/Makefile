#############################################################
vpath  %.c ./
vpath  %.o ../obj

OBJ    = s32pci64-solar.o

#############################################################

all: lib

lib: $(OBJ)
	cd ../obj && ar rv $(LOCAL_LIB) $<

%.o : %.c 
	$(CC) $(CFLAGS) -c $< -o ../obj/$@

dep:
	@if [ ! -e .dep ]; then touch .dep;  fi
	makedepend -Y -f .dep $(IFLAGS) *.c >& /dev/null

clean:
	- rm -f .dep 
	- rm ../obj/*.o
	- ar dv $(LOCAL_LIB) $(OBJ)

#############################################################

-include .dep

