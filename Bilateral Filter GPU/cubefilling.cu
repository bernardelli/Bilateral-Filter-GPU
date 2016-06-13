#include "cubefilling.cuh"


/*
__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x ;
	c[index] = a[index] + b[index];
}
*/

__global__ void cubefilling(const float* image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < image_size.x && j < image_size.y) {
		
		unsigned int k = (unsigned int)image[j + image_size.y*i];
		unsigned int cube_idx = i + image_size.x*j + image_size.x*image_size.y*k;

		dev_cube_wi[cube_idx] = ((float)k);

		dev_cube_w[cube_idx] = 1.0;

		//Next level: perform filling and Z convolution at the same time!
	}


}

void callingCubefilling(const float* dev_image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size)
{

	
	dim3 dimBlock(16, 16);
	dim3 dimGrid((image_size.x + dimBlock.x - 1) / dimBlock.x,
		(image_size.y + dimBlock.y - 1) / dimBlock.y);


	cubefilling <<< dimGrid, dimBlock >>>(dev_image, dev_cube_wi, dev_cube_w, image_size);


}

