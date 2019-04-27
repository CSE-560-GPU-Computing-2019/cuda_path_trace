run:	
	nvcc -I headers/ hope_eternal.cu -o path_trace
	nvprof --system-profiling on -s -f -o profilePT1.nvvp ./path_trace 16 5000 GPUimage1.ppm
	nvprof --system-profiling on -s -f -o profilePT2.nvvp ./path_trace 32 5000 GPUimage2.ppm

clean:	
	rm path_trace *.ppm *.nvvp
