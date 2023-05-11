CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=`pkg-config --libs opencv`
LDLIBS2=-lm -lIL

all: gaussian_blur gaussian_blur-cu

gaussian_blur: gaussian_blur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

gaussian_blur-cu: gaussian_blur.cu
	nvcc -o $@ $<  $(LDLIBS1)


allstreams: gaussian_blur_streams streams

gaussian_blur_streams: gaussian_blur_streams.cu
	nvcc -o $@ $<  $(LDLIBS1)


streams: streams.cu
	nvcc -o $@ $<  $(LDLIBS1)


.PHONY: clean

clean:
	rm gaussian_blur gaussian_blur-cu
