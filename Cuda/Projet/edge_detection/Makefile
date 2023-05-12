CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=`pkg-config --libs opencv`
LDLIBS2=-lm -lIL

all: edge_detection edge_detection-cu edge_detection-streams

edge_detection: edge_detection.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

edge_detection-cu: edge_detection.cu
	nvcc -o $@ $<  $(LDLIBS1)

edge_detection-streams: edge_detection-streams.cu
	nvcc -o $@ $<  $(LDLIBS1)

.PHONY: clean

clean:
	rm edge_detection edge_detection-cu edge_detection-streams
