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
	if (i < dimensions_down.x && j < dimensions_down.y) {
		
		
		//unsigned int cube_idx = floorf((float)i / (float)scale_xy) + dimensions_down.x*floorf((float)j / (float)scale_xy) + dimensions_down.x*dimensions_down.y*floorf((float)k / (float)scale_eps);
		size_t cube_idx_1 = i + dimensions_down.x*j;
#pragma unroll
		for (int ii = 0; ii < scale_xy; ii++)
		{
	#pragma unroll
			for (int jj = 0; jj < scale_xy; jj++)
			{
				size_t i_idx = scale_xy*i + ii;
				size_t j_idx = scale_xy*j + jj;
				if (i_idx < image_size.x && j_idx < image_size.y)
				{
				
					float k = image[i_idx + image_size.x*j_idx];
					size_t cube_idx_2 = cube_idx_1 + dimensions_down.x*dimensions_down.y*floorf(k / (float)scale_eps);
					dev_cube_wi[cube_idx_2] += ((float)k);
					dev_cube_w[cube_idx_2] += 1.0;
				}

			}
		}

		//atomicAdd(&dev_cube_wi[cube_idx], k);
		//atomicAdd(&dev_cube_w[cube_idx], 1.0);
		//dev_cube_wi[cube_idx] += ((float)k);

		//dev_cube_w[cube_idx] += 1.0;

		//Next level: perform filling and Z convolution at the same time!
	}


}

//unused
__global__ void normalize(float *dev_cube_wi, float *dev_cube_w, dim3 dimensions_down)
{
	const int ix = blockDim.x*blockIdx.x + threadIdx.x;
	const int iy = blockDim.y*blockIdx.y + threadIdx.y;
	const int iz = blockIdx.z;
	const int cube_idx = ix + iy*dimensions_down.x + iz*dimensions_down.x*dimensions_down.y;
	float cube_w_value = dev_cube_w[cube_idx];
	if (cube_w_value > 0) {
		dev_cube_w[cube_idx] = 1.0;
		dev_cube_wi[cube_idx] /= cube_w_value;
	}
}

float callingCubefilling(const float* dev_image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{

	
	dim3 dimBlock(16, 16);
	dim3 dimGrid((dimensions_down.x + dimBlock.x - 1) / dimBlock.x,
		(dimensions_down.y + dimBlock.y - 1) / dimBlock.y);

	cudaMemset(dev_cube_wi, 0, dimensions_down.x*dimensions_down.y*dimensions_down.z*sizeof(float)); //seems to be useless
	cudaMemset(dev_cube_w, 0, dimensions_down.x*dimensions_down.y*dimensions_down.z*sizeof(float));
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	
	cubefilling <<< dimGrid, dimBlock >>>(dev_image, dev_cube_wi, dev_cube_w, image_size, scale_xy, scale_eps, dimensions_down);
	cudaDeviceSynchronize();
	
	//const dim3 dimGrid2((dimensions_down.x + dimBlock.x - 1) / dimBlock.x, (dimensions_down.y + dimBlock.y - 1) / dimBlock.y, dimensions_down.z);
	//normalize <<<dimGrid2, dimBlock >>>(dev_cube_wi, dev_cube_w, dimensions_down);
	//cudaDeviceSynchronize();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	
	return time;

}

