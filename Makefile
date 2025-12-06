# Makefile for SSSP project (CPU Dijkstra + GPU Delta-Stepping)

# Directories
SRCDIR := src
BUILDDIR := build
OBJDIR := $(BUILDDIR)/obj
BINDIR := $(BUILDDIR)

# Executable
TARGET := $(BINDIR)/sssp

# Tools
NVCC ?= nvcc

# Flags
CXXFLAGS := -O3 -std=c++11 -I $(SRCDIR)
NVCCFLAGS := -O3 -std=c++11 -I $(SRCDIR) -ccbin g++-11

# Source files
CPP_SRCS := $(SRCDIR)/sssp.cpp \
            $(SRCDIR)/dijkstra.cpp \
            $(SRCDIR)/reader.cpp

CU_SRCS  := $(SRCDIR)/delta.cu

# Object files
CPP_OBJS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SRCS))

OBJS := $(CPP_OBJS) $(CU_OBJS)

.PHONY: all clean run-test0 run-test0-dijkstra run-test0-delta

all: $(TARGET)

# Link final executable with nvcc (handles CUDA runtime automatically)
$(TARGET): $(OBJS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

# Compile C++ sources to objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CUDA sources to objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Ensure build directories exist
$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Run the program
FILE ?= data/small/rome99.gr
METHOD ?= 0
SOURCE ?= 0
TIMED ?= 0

run: $(TARGET)
	$(TARGET) -f $(FILE) -m $(METHOD) -s $(SOURCE) -t $(TIMED)

test: $(TARGET)
	$(TARGET) -f $(FILE) -m 0 -s $(SOURCE) -t $(TIMED) > $(BUILDDIR)/test_output1.txt
	$(TARGET) -f $(FILE) -m 1 -s $(SOURCE) -t $(TIMED) > $(BUILDDIR)/test_output2.txt
	@diff $(BUILDDIR)/test_output1.txt $(BUILDDIR)/test_output2.txt
	@rm -f $(BUILDDIR)/test_output1.txt $(BUILDDIR)/test_output2.txt

time: $(TARGET)
	@echo "Running CPU Dijkstra:"
	@$(TARGET) -f $(FILE) -m 0 -s $(SOURCE) -t 1 > $(BUILDDIR)/test_output1.txt
	@echo "Running GPU Delta-Stepping:"
	@$(TARGET) -f $(FILE) -m 1 -s $(SOURCE) -t 1 > $(BUILDDIR)/test_output2.txt
	@diff $(BUILDDIR)/test_output1.txt $(BUILDDIR)/test_output2.txt
	@rm -f $(BUILDDIR)/test_output1.txt $(BUILDDIR)/test_output2.txt

clean:
	rm -rf $(BUILDDIR)
