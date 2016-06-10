#ifndef LITTLE_CUDA_FUNCTIONS_H
#define LITTLE_CUDA_FUNCTIONS_H

void checkingDevices();

cudaError_t allocateGpuMemory(char **ptr, int size);

cudaError_t copyToGpuMem(float *a, float *b, int size);

#endif