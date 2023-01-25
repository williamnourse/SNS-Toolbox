PROGRAM = main

SRCS = code_objects/synapses_synapses_create_array_codeobject.cu main.cu network.cu objects.cu rand.cu run.cu synapses_classes.cu
H_SRCS = brianlib/clocks.h brianlib/common_math.h brianlib/cudaVector.h brianlib/cuda_utils.h brianlib/curand_buffer.h brianlib/dynamic_array.h brianlib/spikequeue.h brianlib/stdint_compat.h code_objects/synapses_synapses_create_array_codeobject.h network.h objects.h rand.h run.h synapses_classes.h
OBJS = ${SRCS:.cu=.o}
OBJS := ${OBJS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
NVCC = @/usr/local/cuda/bin/nvcc -ccbin $(CXX)
NVCCFLAGS = -I. -std=c++11 -arch=sm_75 -w -use_fast_math  -Xcompiler "-I/home/will/sns-venv/include -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11"
LFLAGS = -lcurand -I. -arch=sm_75 -L/home/will/sns-venv/lib -Xlinker --enable-new-dtags,-R/home/will/sns-venv/lib -w 

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS)
	$(NVCC) $(LFLAGS) $(OBJS) -o $(PROGRAM)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@