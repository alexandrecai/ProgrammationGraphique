CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=`pkg-config --libs opencv`
LDLIBS2=-lm -lIL

all: laplacian_gaussian laplacian_gaussian-cu laplacian_gaussian-streams

laplacian_gaussian: laplacian_gaussian.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

laplacian_gaussian-cu: laplacian_gaussian.cu
	nvcc -o $@ $<  $(LDLIBS1)

laplacian_gaussian-streams: laplacian_gaussian-streams.cu
	nvcc -o $@ $<  $(LDLIBS1)

.PHONY: clean

clean:
	rm laplacian_gaussian laplacian_gaussian-cu laplacian_gaussian-streams
