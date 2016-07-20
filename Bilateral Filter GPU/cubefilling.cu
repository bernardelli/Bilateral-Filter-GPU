#include "cubefilling.cuh"

__global__ void cubefilling_loop(const float* image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < dimensions_down.x && j < dimensions_down.y) {
		
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
					size_t cube_idx_2 = cube_idx_1 + dimensions_down.x*dimensions_down.y*floorf(k / scale_eps);
					dev_cube_wi[cube_idx_2] += k;
					dev_cube_w[cube_idx_2] += 1.0f;
				}

			}
		}
	}


}

__global__ void cubefilling_atomic(const float* image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < image_size.x && j < image_size.y) {
		const float k = image[i + image_size.x*j];
		const size_t cube_idx = (i / scale_xy) + dimensions_down.x*(j / scale_xy) + dimensions_down.x*dimensions_down.y*((int)k / scale_eps);


		atomicAdd(&dev_cube_wi[cube_idx], k);
		atomicAdd(&dev_cube_w[cube_idx], 1.0f);

	}


}



float callingCubefilling(const float* dev_image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{

	
	dim3 dimBlock(16, 16);
	dim3 dimGrid((dimensions_down.x + dimBlock.x - 1) / dimBlock.x,(dimensions_down.y + dimBlock.y - 1) / dimBlock.y); //use for filling_loop

	//dim3 dimGrid((image_size.x + dimBlock.x - 1) / dimBlock.x, (image_size.y + dimBlock.y - 1) / dimBlock.y); //use for filling_atomic

	cudaMemset(dev_cube_wi, 0, dimensions_down.x*dimensions_down.y*dimensions_down.z*sizeof(float)); //seems to be useless
	cudaMemset(dev_cube_w, 0, dimensions_down.x*dimensions_down.y*dimensions_down.z*sizeof(float));




	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	
	cubefilling_loop <<< dimGrid, dimBlock >>>(dev_image, dev_cube_wi, dev_cube_w, image_size, scale_xy, scale_eps, dimensions_down);
	//cubefilling_atomic << < dimGrid, dimBlock >> >(dev_image, dev_cube_wi, dev_cube_w, image_size, scale_xy, scale_eps, dimensions_down);
	cudaDeviceSynchronize();
	
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	
	return time;

}

