NVCC = nvcc

convolution.o: %.o: %.cu %.h include_file.h
convolution_shared.o:  %.o: %.cu %.h include_file.h
cubefilling.o: %.o: %.cu %.h include_file.h
kernel.o:  %.o: %.cu include_file.h 
little_cude_functions.o: %.o: %.cu %.h include_file.h
main.o:  %.o: %.cpp include_file.h
slicing.o:  %.o: %.cu %.h include_file.h

%.o:
	$(NVCC) -c %< -o $@

test: convolution.o convolution_shared.o cubefilling.o kernel.o little_cude_functions.o main.o slicing.o
	$(NVCC) %^ -o $@


