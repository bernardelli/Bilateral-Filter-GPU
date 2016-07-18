#include "convolution_shared.h"

__global__ void convolution_shared_row(float *output, const float *input, const float *kernel, const int kernel_size, const dim3 imsize)
{	
	const int ix = blockDim.x*blockIdx.x + threadIdx.x;
	const int iy = blockDim.y*blockIdx.y + threadIdx.y;
	const int iz = blockIdx.z;

	
	const int cube_idx = ix + iy*imsize.x + iz*imsize.x*imsize.y;

	const int radius_size = kernel_size / 2;

	extern __shared__ float s_image[]; //size is on kernel call, (block_dim_x + 2 * k_radius_xy)*block_dim_y
	const int s_dim_x = blockDim.x + 2 * radius_size;
	const int s_ix = radius_size + threadIdx.x;
	const int s_iy = threadIdx.y;

	if (threadIdx.x < radius_size) //is on the left part of the shared memory!
	{
		s_image[s_ix - radius_size + s_iy*s_dim_x] = 0.0f;
	}
	if (threadIdx.x >(blockDim.x - radius_size))
	{
		s_image[s_ix + radius_size + s_iy*s_dim_x] = 0.0f;
	}

	s_image[s_ix + s_iy*s_dim_x] = (ix >= imsize.x || iy >= imsize.y || iz >= imsize.z) ? 0.0f : input[cube_idx];
	
	__syncthreads();
	float result = 0.0;

#pragma unroll
	for (int i = 0; i < kernel_size; i++)
	{
		result += kernel[i] * s_image[s_ix - i + radius_size + s_iy*s_dim_x];
	}

	if (ix <imsize.x && iy < imsize.y && iz < imsize.z)
	{

		output[cube_idx] = result;
	}


	//if (result > 0 )
	//	printf("%.01f \n", result);
}


__global__ void convolution_shared_col(float *output, const float *input, const float *kernel, const int kernel_size, const dim3 imsize)
{
	const int ix = blockDim.x*blockIdx.x + threadIdx.x;
	const int iy = blockDim.y*blockIdx.y + threadIdx.y;
	const int iz = blockIdx.z;
	const int cube_idx = ix + iy*imsize.x + iz*imsize.x*imsize.y;

	const int radius_size = kernel_size / 2;

	extern __shared__ float s_image__[]; //size is on kernel call, (block_dim_x + 2 * k_radius_xy)*block_dim_y
	const int s_dim_x = blockDim.x;
	const int s_ix = threadIdx.x;
	const int s_iy = radius_size + threadIdx.y;

	if (threadIdx.y < radius_size) //is on the left part of the shared memory!
	{
		s_image__[s_ix + (s_iy - radius_size)*s_dim_x] = 0.0;
	}
	if (threadIdx.y >(blockDim.y - radius_size))
	{
		s_image__[s_ix + (s_iy + radius_size)*s_dim_x] = 0.0;
	}
	s_image__[s_ix + s_iy*s_dim_x] = (ix >= imsize.x || iy >= imsize.y || iz >= imsize.z) ? 0.0f : input[cube_idx];


	__syncthreads();

	float result = 0.0;

#pragma unroll
	for (int i = 0; i < kernel_size; i++)
	{
		result += kernel[i] * s_image__[s_ix + (s_iy - i + radius_size)*s_dim_x];
	}
	if (ix <imsize.x && iy < imsize.y && iz < imsize.z)
	{

		output[cube_idx] = result;
	}

}


__global__ void convolution_shared_eps(float *output, const float *input, const float *kernel, const int kernel_size, const dim3 imsize)
{
	const int iz = blockDim.x*blockIdx.x + threadIdx.x;
	const int ix = blockDim.y*blockIdx.y + threadIdx.y;
	const int iy = blockIdx.z;
	
	const int cube_idx = ix + iy*imsize.x + iz*imsize.x*imsize.y;

	const int radius_size = kernel_size / 2;

	extern __shared__ float s_image_[]; //size is on kernel call, (block_dim_x + 2 * k_radius_xy)*block_dim_y
	const int s_dim_x = blockDim.x + 2 * radius_size;
	const int s_ix = radius_size + threadIdx.x;
	const int s_iy = threadIdx.y;

	if (threadIdx.x < radius_size) //is on the left part of the shared memory!
	{
		s_image_[s_ix - radius_size + s_iy*s_dim_x] = 0.0;
	}
	if (threadIdx.x >(blockDim.x - radius_size))
	{
		s_image_[s_ix + radius_size + s_iy*s_dim_x] = 0.0;
	}
	s_image_[s_ix + s_iy*s_dim_x] = (ix >= imsize.x || iy >= imsize.y || iz >= imsize.z) ? 0.0f : input[cube_idx];



	__syncthreads();

	float result = 0.0;

#pragma unroll
	for (int i = 0; i < kernel_size; i++)
	{
		result += kernel[i] * s_image_[s_ix - i + radius_size + s_iy*s_dim_x];
	}
	if (ix <imsize.x && iy < imsize.y && iz < imsize.z)
	{

		output[cube_idx] = result;
	}
	

	//if (result > 0 )
		//printf("%.01f \n", result);
}


float callingConvolution_shared(float *dev_cube_wi_out, float *dev_cube_w_out, float *dev_cube_wi, float *dev_cube_w, const float *dev_kernel_xy, int kernel_xy_size, const float *dev_kernel_eps, int kernel_eps_size, dim3  image_dimensions, int device)
{
	/**Getting shared memory size and max block size 
	*/

	//TODO: intitialize this on main
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device); 



	int max_shared_mem = deviceProp.sharedMemPerBlock / sizeof(float);
	int k_radius_xy = kernel_xy_size / 2;
	//Conv row

	//deviceProp.maxThreadsPerMultiProcessor;
	//deviceProp.sharedMemPerMultiprocessor;


	int block_dim_x = image_dimensions.x; //make this work later for big images
	int block_dim_y = max_shared_mem / (block_dim_x + 2 * k_radius_xy);

	if (block_dim_x*block_dim_y > deviceProp.maxThreadsPerBlock)
	{
		block_dim_y = deviceProp.maxThreadsPerBlock / block_dim_x;
	}
	
	int shared_memory_size = sizeof(float)*(block_dim_x + 2 * k_radius_xy)*block_dim_y;


	const dim3 block(block_dim_x, block_dim_y); //threads per block 32 32
	const dim3 grid((image_dimensions.x + block_dim_x - 1) / block_dim_x, (image_dimensions.y + block_dim_y - 1) / block_dim_y, image_dimensions.z);
	
	cudaEvent_t start_row_1, stop_row_1, start_row_2, stop_row_2;
	float time_shared_row_1, time_shared_row_2;
	cudaEventCreate(&start_row_1);
	cudaEventCreate(&stop_row_1);
	cudaEventRecord(start_row_1);
	
	convolution_shared_row <<< grid, block, shared_memory_size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_xy, kernel_xy_size, image_dimensions);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_row_1);
	cudaEventSynchronize(stop_row_1);
	cudaEventElapsedTime(&time_shared_row_1, start_row_1, stop_row_1);
	
	swap2(&dev_cube_wi_out, &dev_cube_wi);
	
	cudaEventCreate(&start_row_2);
	cudaEventCreate(&stop_row_2);
	cudaEventRecord(start_row_2);
	
	convolution_shared_row <<< grid, block, shared_memory_size >>>(dev_cube_w_out, dev_cube_w, dev_kernel_xy, kernel_xy_size, image_dimensions);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_row_2);
	cudaEventSynchronize(stop_row_2);
	cudaEventElapsedTime(&time_shared_row_2, start_row_2, stop_row_2);
	
	swap2(&dev_cube_w_out, &dev_cube_w);

	//Conv Col



	block_dim_y = image_dimensions.y; //make this work later for big images
	block_dim_x = max_shared_mem / (block_dim_y + 2 * k_radius_xy);

	if (block_dim_x*block_dim_y > deviceProp.maxThreadsPerBlock)
	{
		block_dim_x = deviceProp.maxThreadsPerBlock / block_dim_y;
	}

	shared_memory_size = sizeof(float)*block_dim_x*(block_dim_y + 2 * k_radius_xy);


	const dim3 block2(block_dim_x, block_dim_y); //threads per block 32 32
	const dim3 grid2((image_dimensions.x + block_dim_x - 1) / block_dim_x, (image_dimensions.y + block_dim_y - 1) / block_dim_y, image_dimensions.z);
	
	cudaEvent_t start_col_1, stop_col_1, start_col_2, stop_col_2;
	float time_shared_col_1, time_shared_col_2;
	cudaEventCreate(&start_col_1);
	cudaEventCreate(&stop_col_1);
	cudaEventRecord(start_col_1);
	
	convolution_shared_col <<< grid2, block2, shared_memory_size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_xy, kernel_xy_size, image_dimensions);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_col_1);
	cudaEventSynchronize(stop_col_1);
	cudaEventElapsedTime(&time_shared_col_1, start_col_1, stop_col_1);
	
	swap2(&dev_cube_wi_out, &dev_cube_wi);
	
	cudaEventCreate(&start_col_2);
	cudaEventCreate(&stop_col_2);
	cudaEventRecord(start_col_2);
	
	convolution_shared_col <<< grid2, block2, shared_memory_size >>>(dev_cube_w_out, dev_cube_w, dev_kernel_xy, kernel_xy_size, image_dimensions);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_col_2);
	cudaEventSynchronize(stop_col_2);
	cudaEventElapsedTime(&time_shared_col_2, start_col_2, stop_col_2);
	
	swap2(&dev_cube_w_out, &dev_cube_w);
	// conv eps

	//Conv Col

	int k_radius_eps = kernel_eps_size / 2;

	int block_dim_eps = image_dimensions.z; //make this work later for big images
	block_dim_x = max_shared_mem / (block_dim_eps + 2 * k_radius_eps);

	if (block_dim_eps*block_dim_x > deviceProp.maxThreadsPerBlock)
	{
		block_dim_x = deviceProp.maxThreadsPerBlock / block_dim_eps;
	}

	shared_memory_size = sizeof(float)*(block_dim_eps + 2 * k_radius_eps)*(block_dim_x);


	const dim3 block3(block_dim_eps, block_dim_x); //threads per block 32 32
	const dim3 grid3((image_dimensions.z + block_dim_eps - 1) / block_dim_eps, (image_dimensions.x + block_dim_x - 1) / block_dim_x, image_dimensions.y);

	cudaEvent_t start_eps_1, stop_eps_1, start_eps_2, stop_eps_2;
	float time_shared_eps_1, time_shared_eps_2;
	cudaEventCreate(&start_eps_1);
	cudaEventCreate(&stop_eps_1);
	cudaEventRecord(start_eps_1);
	
	convolution_shared_eps <<< grid3, block3, shared_memory_size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_eps, kernel_eps_size, image_dimensions);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_eps_1);
	cudaEventSynchronize(stop_eps_1);
	cudaEventElapsedTime(&time_shared_eps_1, start_eps_1, stop_eps_1);
	
	cudaEventCreate(&start_eps_2);
	cudaEventCreate(&stop_eps_2);
	cudaEventRecord(start_eps_2);
	
	convolution_shared_eps <<< grid3, block3, shared_memory_size >>>(dev_cube_w_out, dev_cube_w, dev_kernel_eps, kernel_eps_size, image_dimensions);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_eps_2);
	cudaEventSynchronize(stop_eps_2);
	cudaEventElapsedTime(&time_shared_eps_2, start_eps_2, stop_eps_2);
	float time = time_shared_row_1 + time_shared_row_2 + time_shared_col_1 + time_shared_col_2 + time_shared_eps_1 + time_shared_eps_2;
	return time;
}

void swap2(float** a, float** b){
	float* c = *a;
    *a = *b;
    *b = c;
}