run:	
	g++ -fopenmp pt.cpp -o pt_test 
	nvcc -I headers/ hope_eternal.cu -o path_trace
	nvprof --print-gpu-summary -f -o profilePT.nvvp ./path_trace
	time ./pt_test 5000
clean:	
	rm *.ppm path_trace *.nvvp pt_test
