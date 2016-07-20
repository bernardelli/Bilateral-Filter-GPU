#include "convolution_sep.cuh"

/*Performs separable convolution on 3d cube*/

__global__ void convolution_sep(float *output, const float *input, const float *kernel, const int kernel_size, const dim3 imsize, int dir)
{
	size_t ix, iy, iz;
	if (dir == X_DIR)
	{
		ix = blockDim.x*blockIdx.x + threadIdx.x;
		iy = blockDim.y*blockIdx.y + threadIdx.y;
		iz = blockIdx.z;
	}
	else if (dir == Y_DIR)
	{
		iy = blockDim.x*blockIdx.x + threadIdx.x;
		ix = blockDim.y*blockIdx.y + threadIdx.y;
		iz = blockIdx.z;
	}
	else if (dir == EPS_DIR)
	{
		iz = blockDim.x*blockIdx.x + threadIdx.x;
		ix = blockDim.y*blockIdx.y + threadIdx.y;
		iy = blockIdx.z;
	}

	const bool valid = ix < imsize.x && iy < imsize.y && iz < imsize.z;
	const size_t cube_idx = ix + iy*imsize.x + iz*imsize.x*imsize.y;

	const size_t radius_size = kernel_size / 2;

	extern __shared__ float s_image[]; //size is on kernel call
	const size_t s_dim_x = blockDim.x + 2 * radius_size;
	const size_t s_ix = radius_size + threadIdx.x;
	const size_t s_iy = threadIdx.y;
	float result = 0.0;

	if (threadIdx.x < radius_size) //is on the left part of the shared memory
	{
		s_image[s_ix - radius_size + s_iy*s_dim_x] = 0.0f;
	}
	if (threadIdx.x >= (blockDim.x - radius_size)) //is on the right part
	{
		s_image[s_ix + radius_size + s_iy*s_dim_x] = 0.0f;
	}

	

	s_image[s_ix + s_iy*s_dim_x] = (valid) ? input[cube_idx] : 0.0f;


	__syncthreads();


#pragma unroll
	for (int i = 0; i < kernel_size; i++)
	{
		result += kernel[i] * s_image[s_ix - i + radius_size + s_iy*s_dim_x];
	}

	if (valid)
	{

		output[cube_idx] = result;
	}
}


float callingConvolution_sep(float *dev_cube_wi_out, float *dev_cube_w_out, float *dev_cube_wi, float *dev_cube_w, const float *dev_kernel_xy, int kernel_xy_size, const float *dev_kernel_eps, int kernel_eps_size, dim3  image_dimensions, int device)
{



	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device); 


	/*Convolution on row*/

	size_t max_shared_mem = deviceProp.sharedMemPerBlock / sizeof(float);
	int k_radius_xy = kernel_xy_size / 2;


	//finds most appropriate block size for a given gpu
	size_t block_dim_x = image_dimensions.x;
	size_t block_dim_y = max_shared_mem / (block_dim_x + 2 * k_radius_xy);

	if (block_dim_x*block_dim_y > deviceProp.maxThreadsPerBlock)
	{
		block_dim_y = deviceProp.maxThreadsPerBlock / block_dim_x;
	}
	if (block_dim_y > image_dimensions.y) {
		block_dim_y = image_dimensions.y;
	}
	
	size_t shared_memory_size = sizeof(float)*(block_dim_x + 2 * k_radius_xy)*block_dim_y;


	dim3 blockx(block_dim_x, block_dim_y);
	dim3 gridx((image_dimensions.x + block_dim_x - 1) / block_dim_x, (image_dimensions.y + block_dim_y - 1) / block_dim_y, image_dimensions.z);
	
	cudaEvent_t start_row_1, stop_row_1, start_row_2, stop_row_2;
	float time_shared_row_1, time_shared_row_2;
	cudaEventCreate(&start_row_1);
	cudaEventCreate(&stop_row_1);
	cudaEventRecord(start_row_1);
	
	convolution_sep <<< gridx, blockx, shared_memory_size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_xy, kernel_xy_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_row_1);
	cudaEventSynchronize(stop_row_1);
	cudaEventElapsedTime(&time_shared_row_1, start_row_1, stop_row_1);
	
	swap2(&dev_cube_wi_out, &dev_cube_wi);
	
	cudaEventCreate(&start_row_2);
	cudaEventCreate(&stop_row_2);
	cudaEventRecord(start_row_2);
	
	convolution_sep <<< gridx, blockx, shared_memory_size >> >(dev_cube_w_out, dev_cube_w, dev_kernel_xy, kernel_xy_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_row_2);
	cudaEventSynchronize(stop_row_2);
	cudaEventElapsedTime(&time_shared_row_2, start_row_2, stop_row_2);
	
	swap2(&dev_cube_w_out, &dev_cube_w);

	/*Convolution on column*/


	//finds most appropriate block size for a given gpu
	block_dim_y = image_dimensions.y; 
	block_dim_x = max_shared_mem / (block_dim_y + 2 * k_radius_xy);

	if (block_dim_x*block_dim_y > deviceProp.maxThreadsPerBlock)
	{
		block_dim_x = deviceProp.maxThreadsPerBlock / block_dim_y;
	}

	if (block_dim_x > image_dimensions.x) {
		block_dim_x = image_dimensions.x;
	}

	shared_memory_size = sizeof(float)*block_dim_x*(block_dim_y + 2 * k_radius_xy);


	 dim3 blocky(block_dim_y, block_dim_x); 
	 dim3 gridy((image_dimensions.y + block_dim_y - 1) / block_dim_y, (image_dimensions.x + block_dim_x - 1) / block_dim_x, image_dimensions.z);
	
	cudaEvent_t start_col_1, stop_col_1, start_col_2, stop_col_2;
	float time_shared_col_1, time_shared_col_2;
	cudaEventCreate(&start_col_1);
	cudaEventCreate(&stop_col_1);
	cudaEventRecord(start_col_1);
	
	convolution_sep <<< gridy, blocky, shared_memory_size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_xy, kernel_xy_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_col_1);
	cudaEventSynchronize(stop_col_1);
	cudaEventElapsedTime(&time_shared_col_1, start_col_1, stop_col_1);
	
	swap2(&dev_cube_wi_out, &dev_cube_wi);
	
	cudaEventCreate(&start_col_2);
	cudaEventCreate(&stop_col_2);
	cudaEventRecord(start_col_2);
	
	convolution_sep <<< gridy, blocky, shared_memory_size >>>(dev_cube_w_out, dev_cube_w, dev_kernel_xy, kernel_xy_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_col_2);
	cudaEventSynchronize(stop_col_2);
	cudaEventElapsedTime(&time_shared_col_2, start_col_2, stop_col_2);
	
	swap2(&dev_cube_w_out, &dev_cube_w);

	/*Convolution on range*/

	//finds most appropriate block size for a given gpu
	int k_radius_eps = kernel_eps_size / 2;

	int block_dim_eps = image_dimensions.z;
	block_dim_x = max_shared_mem / (block_dim_eps + 2 * k_radius_eps);

	if (block_dim_eps*block_dim_x > deviceProp.maxThreadsPerBlock)
	{
		block_dim_x = deviceProp.maxThreadsPerBlock / block_dim_eps;
	}
	if (block_dim_x > image_dimensions.x) {
		block_dim_x = image_dimensions.x;
	}

	shared_memory_size = sizeof(float)*(block_dim_eps + 2 * k_radius_eps)*(block_dim_x);


	const dim3 blockeps(block_dim_eps, block_dim_x); //threads per block 32 32
	const dim3 grideps((image_dimensions.z + block_dim_eps - 1) / block_dim_eps, (image_dimensions.x + block_dim_x - 1) / block_dim_x, image_dimensions.y);

	cudaEvent_t start_eps_1, stop_eps_1, start_eps_2, stop_eps_2;
	float time_shared_eps_1, time_shared_eps_2;
	cudaEventCreate(&start_eps_1);
	cudaEventCreate(&stop_eps_1);
	cudaEventRecord(start_eps_1);
	
	convolution_sep <<< grideps, blockeps, shared_memory_size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_eps, kernel_eps_size, image_dimensions, EPS_DIR);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop_eps_1);
	cudaEventSynchronize(stop_eps_1);
	cudaEventElapsedTime(&time_shared_eps_1, start_eps_1, stop_eps_1);
	
	cudaEventCreate(&start_eps_2);
	cudaEventCreate(&stop_eps_2);
	cudaEventRecord(start_eps_2);
	
	convolution_sep <<< grideps, blockeps, shared_memory_size >>>(dev_cube_w_out, dev_cube_w, dev_kernel_eps, kernel_eps_size, image_dimensions, EPS_DIR);
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