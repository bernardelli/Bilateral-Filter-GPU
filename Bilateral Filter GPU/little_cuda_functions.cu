#include "little_cuda_functions.h"

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

	//TESTING
	int kernel_xy_size = 25;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // device = 0;

	int max_shared_mem = deviceProp.sharedMemPerBlock / sizeof(float);

	//deviceProp.maxThreadsPerMultiProcessor;
	//deviceProp.sharedMemPerMultiprocessor;
	int k_radius_xy = kernel_xy_size / 2;
	int regular_block_x_dim = 32;
	int block_dim_x = (k_radius_xy > regular_block_x_dim) ? k_radius_xy : regular_block_x_dim;
	int block_dim_y = max_shared_mem / (block_dim_x + 2 * k_radius_xy);

	if (block_dim_x*block_dim_y > deviceProp.maxThreadsPerBlock)
	{
		block_dim_y = deviceProp.maxThreadsPerBlock / block_dim_x;
	}

	
}

cudaError_t allocateGpuMemory(float**ptr, int size)
{
	cudaError_t cudaStatus = cudaMalloc((float**)ptr, size * sizeof(float));
	return cudaStatus;
}

cudaError_t copyToGpuMem(float *a, float *b, int size)
{
	cudaError_t cudaStatus = cudaMemcpy(a, b, size * sizeof(float), cudaMemcpyHostToDevice);
	return cudaStatus;
}