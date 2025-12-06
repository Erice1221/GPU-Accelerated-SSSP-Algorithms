sequential:
	g++ -std=c++20 -O3 sequential.cpp -o sequential

EXECUTABLE := parallel
CU_FILES   := parallel.cu parallel_frontier.cu
CC_FILES   := main.cpp

OBJDIR=objs
CXX=g++-11 -m64
CXXFLAGS=-O3 -Wall 
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/g++-11

OBJS=$(OBJDIR)/main.o $(OBJDIR)/parallel.o $(OBJDIR)/parallel_frontier.o

.PHONY: dirs clean parallel sequential

default: parallel

parallel: dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDFLAGS)

dirs:
	mkdir -p $(OBJDIR)/

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

$(OBJDIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJDIR) sequential $(EXECUTABLE)