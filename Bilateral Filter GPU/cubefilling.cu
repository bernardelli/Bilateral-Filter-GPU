#include "cubefilling.cuh"


/*
__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x ;
	c[index] = a[index] + b[index];
}
*/

__global__ void cubefilling(const float* image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < image_size.x && j < image_size.y) {
		
		unsigned int k = (unsigned int)image[j + image_size.y*i];
		unsigned int cube_idx = floorf(i/scale_xy) + dimensions_down.x*floorf(j/scale_xy) + dimensions_down.x*dimensions_down.y*floorf(k/scale_eps);
		

		atomicAdd(&dev_cube_wi[cube_idx], (float) k);
		atomicAdd(&dev_cube_w[cube_idx], 1.0);
		//dev_cube_wi[cube_idx] += ((float)k);

		//dev_cube_w[cube_idx] += 1.0;

		//Next level: perform filling and Z convolution at the same time!
	}


}

float callingCubefilling(const float* dev_image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{

	
	dim3 dimBlock(16, 16);
	dim3 dimGrid((image_size.x + dimBlock.x - 1) / dimBlock.x,
		(image_size.y + dimBlock.y - 1) / dimBlock.y);

	//cudaMemset(dev_cube_wi, 0, image_size.x*image_size.y*image_size.z*sizeof(float)); //seems to be useless
	//cudaMemset(dev_cube_w, 0, image_size.x*image_size.y*image_size.z*sizeof(float));
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	
	cubefilling <<< dimGrid, dimBlock >>>(dev_image, dev_cube_wi, dev_cube_w, image_size, scale_xy, scale_eps, dimensions_down);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	
	return time;

}

