CPP=g++
RM=rm -f
CPPFLAGS=-Wall -I./include
LDFLAGS=
LDLIBS=-lOpenCL
EXE=openclTest

SRCS=$(shell ls src/*.cpp)
OBJS=$(subst src/,obj/,$(subst .cpp,.o,$(SRCS)))

all: $(EXE)
	@echo
	@echo
	@echo \*\*\* SUCCESS!! ./$(EXE) is ready to go. \*\*\*
	@echo

openclTest: $(OBJS)
	$(CPP) $(LDFLAGS) -o $(EXE) $(OBJS) $(LDLIBS)

obj/%.o: src/%.cpp
	$(CPP) $(CPPFLAGS) -c -o $@ $^

clean:
	$(RM) $(OBJS) $(EXE)