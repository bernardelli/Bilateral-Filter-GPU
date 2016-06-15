#include "convolution_shared.h"

__global__ void convolution__shared_row(float *output, const float *input, const float *kernel, const int ksize, const dim3 imsize)
{	
	unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;
	unsigned int i = ix + iy*blockDim.x*gridDim.x;
	//unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int imsize_x = imsize.x;
	unsigned int imsize_y = imsize.y;
	unsigned int imsize_z = imsize.z;
	unsigned int im_size = imsize_x*imsize_y*imsize_z;
	//printf("i = %d\n", i);
	//idx = x_i + imsize_x*y_i + imsize_x*imsize_y*z_i
	unsigned int z_i = i / (unsigned int) (imsize_x*imsize_y);
	unsigned int y_i = (i - z_i*(unsigned int)(imsize_x*imsize_y)) / imsize_x;
	unsigned int x_i = i - y_i*imsize_x - z_i*(unsigned int)(imsize_x*imsize_y);

	__shared__ float s_image[BLOCK_DIM][BLOCK_DIM];
	
	/*
	double result = 0.0;
	unsigned int idx = 0;
	unsigned int k_offset = (ksize / 2);
	for (int k = 0; k < ksize; k++) {
		if (dir == X_DIR) {
			int x_input = k_offset - k + x_i;
			if (x_input >= 0 && x_input < imsize_x) {
				idx = (unsigned int)x_input + imsize_x*y_i + imsize_x*imsize_y*z_i;
				if (idx < im_size)
					result += input[idx]*kernel[k];
			}
		}
		else if (dir == Y_DIR) {
			int y_input = k_offset - k + y_i;

			if (y_input >= 0 && y_input < imsize_y) {
				idx = x_i + imsize_x*(unsigned int)y_input + imsize_x*imsize_y*z_i;
				if (idx < im_size)
					result += input[idx] * kernel[k];
			}
			//else
				//printf("out of bounds\n");
		}
		else if (dir == Z_DIR) {
			int z_input = k_offset - k + z_i;
			//printf("z_input %f\n", z_input);
			if (z_input >= 0 && z_input < imsize_z) {
				idx = x_i + imsize_x*y_i + imsize_x*imsize_y*(unsigned int) z_input;
				if (idx < im_size)
					result += input[idx] * kernel[k];
			}
			//else
				//printf("out of bounds\n");
		}
		else
			printf("All wrong");
		
	}
	idx = x_i + imsize_x*y_i + imsize_x*imsize_y*z_i;
	if (idx < im_size)
		output[idx] = result;
		*/
}

void callingConvolution_shared(float *dev_cube_wi_out, float *dev_cube_w_out, float *dev_cube_wi, float *dev_cube_w, const float *dev_kernel_xy, int kernel_xy_size, const float *dev_kernel_eps, int kernel_eps_size, dim3  image_dimensions)
{
	/**Getting shared memory size and max block size 
	*/
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // device = 0;

	int max_shared_mem = deviceProp.sharedMemPerBlock/sizeof(float);
	
	//deviceProp.maxThreadsPerMultiProcessor;
	//deviceProp.sharedMemPerMultiprocessor;
	int k_radius_xy = kernel_xy_size / 2;
	int regular_block_x_dim = 32;
	int block_dim_x = (k_radius_xy > regular_block_x_dim) ? k_radius_xy : regular_block_x_dim;
	int block_dim_y = max_shared_mem / (block_dim_x + 2 * k_radius_xy);

	if (block_dim_x*block_dim_y > deviceProp.maxThreadsPerBlock)
	{
		block_dim_y = block_dim_x / deviceProp.maxThreadsPerBlock;
	}



	const dim3 block(BLOCK_DIM, BLOCK_DIM); //threads per block 32 32

	int grin = 256;
	const dim3 grid(grin, grin);
	/*
	
	convolution <<< grid, block >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_xy, kernel_xy_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	swap(&dev_cube_wi, &dev_cube_wi_out);

	convolution <<< grid, block >>>(dev_cube_w_out, dev_cube_w, dev_kernel_xy, kernel_xy_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	swap(&dev_cube_w, &dev_cube_w_out);
		
	convolution <<< grid, block >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_xy, kernel_xy_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	swap(&dev_cube_wi, &dev_cube_wi_out);
	
	convolution <<< grid, block >>>(dev_cube_w_out, dev_cube_w, dev_kernel_xy, kernel_xy_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	swap(&dev_cube_w, &dev_cube_w_out);
	
	convolution <<< grid, block >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel_eps, kernel_eps_size, image_dimensions, Z_DIR);
	cudaDeviceSynchronize();
	swap(&dev_cube_wi, &dev_cube_wi_out);
	
	convolution <<< grid, block >>>(dev_cube_w_out, dev_cube_w, dev_kernel_eps, kernel_eps_size, image_dimensions, Z_DIR);
	cudaDeviceSynchronize();
	swap(&dev_cube_w, &dev_cube_w_out);*/
	
}
/*
void swap(float** a, float** b){
	float* c = *a;
    *a = *b;
    *b = c;
	}*/