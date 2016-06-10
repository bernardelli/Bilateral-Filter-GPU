
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "includes.h"

#define X_DIR 0
#define Y_DIR 1
#define Z_DIR 2

//TODO solve shit with grid/block dimentions access of cube memory
//http://stackoverflow.com/questions/21971484/calculation-on-gpu-leads-to-driver-error-stopped-responding
void define_kernel(float* output_kernel, float sigma, int size);

__global__ void slicing(float *dev_image, const float*dev_cube_wi, const float*dev_cube_w, const dim3 imsize)
{



	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((i < imsize.x) && (j < imsize.y))
	{

		int value = (int)dev_image[j + imsize.y*i];
		//printf("value = %d, i = %d, j = %d\n", value, i, j);

		unsigned int cube_idx = i + imsize.x*j + imsize.x*imsize.y*value;
		dev_image[j + imsize.y*i] = dev_cube_wi[cube_idx] / dev_cube_w[cube_idx];
		//printf("w = %f  wi = %f \n", dev_cube_w[cube_idx], dev_cube_wi[cube_idx]);
		
	}
	//else
		//printf("out of bounds\n");

}
__global__ void convolution(float *output, const float *input, const float* kernel, const int ksize,  const dim3 imsize, const int dir)
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
			//else
				//printf("out of bounds\n");
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
}


int main(int argc, char **argv)
{

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d and concurrentKernels = %d.\n",
			device, deviceProp.major, deviceProp.minor, deviceProp.concurrentKernels);
	}
	cudaDeviceReset();

	//Load Image
	
	cv::Mat image;
	image = cv::imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Original image", image);

	//Set up cubes
	int size = image.rows*image.cols * 256;
	float *cube_wi, *cube_w;
	int kernel_size = 57;
	float *kernel = (float*)malloc(kernel_size*sizeof(float));
	define_kernel(kernel, 25.5, kernel_size);


	float *dev_cube_wi, *dev_cube_w, *dev_cube_wi_out, *dev_cube_w_out, *dev_kernel;
	cube_wi = (float*)calloc(size, sizeof(float));
	cube_w = (float*)calloc(size, sizeof(float));
	//filling //PERFORM FILLING WITH CUDA KERNEL
			//later try filling and doing z-direction conv. at once!
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			
			unsigned int k = image.at<uchar>(i, j);
			cube_wi[i + image.rows*j + image.rows*image.cols*k] = ((float)k );
			//std::cout << k << std::endl;
			cube_w[i + image.rows*j + image.rows*image.cols*k] = 1.0;
					//std::cout << "assigned" << std::endl;
				
			
		}
	}
	


	dim3  image_dimensions = dim3(image.rows, image.cols, 256);
	

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cube_wi, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_cube_w, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cube_wi_out, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cube_w_out, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_kernel, kernel_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}


	cudaStatus = cudaMemcpy(dev_cube_wi, cube_wi, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(dev_cube_w, cube_w, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(dev_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	// block 64x64 and grid 128x128 or block 32 32 and grid 256 256
	//ITS WRONG BUT WORKS FOR NOT A LOT OF THREADS IDK
	const dim3 block(32, 32); //threads per block 32 32

	//Calculate grid size to cover the whole image
	//int grid_n = ceil(sqrt(size) / (block.y));
	int grin = 256;//256 made it work with 231
	const dim3 grid(grin, grin); //number of blocks
	//int block = 1024;
	//int grid = size / block;

	convolution << <grid, block >> >(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	float * temp;
	temp = dev_cube_wi;
	dev_cube_wi = dev_cube_wi_out;
	dev_cube_wi_out = temp;
	

	convolution << <grid, block >> >(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	temp = dev_cube_w;
	dev_cube_w = dev_cube_w_out;
	dev_cube_w_out = temp;
	
	convolution << <grid, block >> >(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	temp = dev_cube_wi;
	dev_cube_wi = dev_cube_wi_out;
	dev_cube_wi_out = temp;
	


	convolution << <grid, block >> >(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	temp = dev_cube_w;
	dev_cube_w = dev_cube_w_out;
	dev_cube_w_out = temp;
	

	
	convolution << <grid, block >> >(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, Z_DIR);
	cudaDeviceSynchronize();
	
	temp = dev_cube_wi;
	dev_cube_wi = dev_cube_wi_out;
	dev_cube_wi_out = temp;
	

	convolution << <grid, block >> >(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, Z_DIR);
	cudaDeviceSynchronize();
	temp = dev_cube_w;
	dev_cube_w = dev_cube_w_out;
	dev_cube_w_out = temp;
	
	//allocate gpu image
	float * dev_image, *result_image;

	image.convertTo(image, CV_32F);
	int imsize = image.rows*image.cols;
	cudaMalloc(&dev_image, imsize*sizeof(float));
	cudaMemcpy(dev_image, image.ptr(), imsize*sizeof(float), cudaMemcpyHostToDevice);
	


	//Specify a reasonable block size
	const dim3 block2(32, 32);

	//Calculate grid size to cover the whole image
	const dim3 grid2(((image.cols + block2.x - 1) / block2.x), ((image.rows + block2.y - 1) / block2.y));

	slicing << < grid2, block2 >> > (dev_image , dev_cube_wi, dev_cube_w, image_dimensions);
	cudaDeviceSynchronize();

	result_image = (float*)malloc(imsize*sizeof(float));
	cudaMemcpy(result_image, dev_image, imsize*sizeof(float), cudaMemcpyDeviceToHost);


	//cv::gpu::GpuMat dev_image(image.rows, image.cols, CV_8U, dev_image);

	cv::Mat output_imag(image.rows, image.cols, CV_32F, result_image);



	
	//std::cout << output_imag << std::endl;
	//dev_output_image.download(output_imag);

	cudaFree(dev_cube_wi_out);
	cudaFree(dev_cube_wi);

	cudaFree(dev_cube_w_out);
	cudaFree(dev_cube_w);
	cudaFree(dev_image);

	cudaFree(dev_kernel);
	free(cube_w);
	free(cube_wi);
	free(kernel);

	cv::namedWindow("Filtered image", cv::WINDOW_AUTOSIZE);// Create a window for display.

	cv::imshow("Filtered image", output_imag/256);
	cv::imwrite("Result.bmp", output_imag);
	cv::waitKey(0);



   

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


void define_kernel(float* output_kernel, float sigma, int size) {
	for (int i = 0; i < size; i++) {
		output_kernel[i] = expf(-0.5*powf((size / 2 - i) / sigma, 2));
	}
}