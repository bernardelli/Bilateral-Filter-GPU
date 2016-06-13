#include "little_cuda_functions.h"


/********************************************************************************
*** checking if CUDA is installed and printing out the compute capability and ***
*** the concurrent kernels of each device                                     ***
********************************************************************************/
void checkingDevices()
{
	int deviceCount;
	cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
	
	if (cudaStatus == cudaErrorInsufficientDriver) {
		fprintf(stderr, "cudaGetDeviceCount failed!  Do you have CUDA installed?");
	}
	
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d and concurrentKernels = %d.\n",
			device, deviceProp.major, deviceProp.minor, deviceProp.concurrentKernels);
	}
	cudaDeviceReset();
}


/********************************************************************************
*** getting the size of the malloced space and returning a pointer of that    ***
*** malloced space                                                            ***
********************************************************************************/
float* allocateGpuMemory(int size)
{

	float* p;
	size *= sizeof(float);
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&p, size);

	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Error code: %s\n", cudaGetErrorString);
		return NULL;
	}

	return p;
}


/********************************************************************************
*** copy memory to the gpu memory                                             ***
********************************************************************************/
cudaError_t copyToGpuMem(float *a, float *b, int size)
{
	cudaError_t cudaStatus = cudaMemcpy(a, b, size * sizeof(float), cudaMemcpyHostToDevice);
	return cudaStatus;
}