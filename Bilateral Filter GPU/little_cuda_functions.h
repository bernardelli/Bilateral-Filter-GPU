#ifndef LITTLE_CUDA_FUNCTIONS_H
#define LITTLE_CUDA_FUNCTIONS_H

#include "include_file.h"


/********************************************************************************
*** checking if CUDA is installed and printing out the compute capability and ***
*** the concurrent kernels of each device                                     ***
********************************************************************************/
void checkingDevices();


/********************************************************************************
*** getting the size of the malloced space and returning a pointer of that    ***
*** malloced space                                                            ***
********************************************************************************/
float* allocateGpuMemory(int size);


/********************************************************************************
*** copy memory to the gpu memory                                             ***
********************************************************************************/
cudaError_t copyToGpuMem(float *a, float *b, int size);

#endif