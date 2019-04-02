run:	
#	g++ -fopenmp pt.cpp -o pt_test 
	nvcc -I headers/ hope_eternal.cu -o path_trace
	nvprof --system-profiling on -s -f -o profilePT.nvvp ./path_trace
#	time ./pt_test 5000
clean:	
	rm *.ppm path_trace *.nvvp
#	rm pt_test
