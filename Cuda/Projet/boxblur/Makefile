CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=`pkg-config --libs opencv`
LDLIBS2=-lm -lIL

all: boxblur boxblur-cu boxblur-streams

boxblur: boxblur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

boxblur-cu: boxblur.cu
	nvcc -o $@ $<  $(LDLIBS1)

boxblur-streams: boxblur-streams.cu
	nvcc -o $@ $<  $(LDLIBS1)

.PHONY: clean

clean:
	rm boxblur boxblur-cu boxblur-streams
