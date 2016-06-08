
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


const static char *sSDKsample = "CUDA Bilateral Filter";

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void define_kernel(float* output_kernel, float sigma, int size);

__global__ void slicing (float *dev_image, const float*dev_cube_wi, const float*dev_cube_w, const dim3 imsize)
{



	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((i < imsize.x) && (j < imsize.y))
	{

		int value = (int)dev_image[i + imsize.x*j];
		//printf("value = %d, i = %d, j = %d\n", value, i, j);

		int cube_idx = i + imsize.x*j + imsize.x*imsize.y*value;
		dev_image[i + imsize.x*j] = dev_cube_wi[cube_idx] / dev_cube_w[cube_idx];

	}
	else
		printf("out of bounds\n");

}
__global__ void convolution(float *output, const float *input, const float* kernel, const int ksize,  const dim3 imsize, const int dir)
{
	int i = threadIdx.x;
	//idx = x_i + imsize.x*y_i + imsize.x*imsize.y*z_i
	int z_i = i / (imsize.x*imsize.y);
	int y_i = (i - z_i*(imsize.x*imsize.y)) / imsize.x;
	int x_i = i -y_i*imsize.x- z_i*(imsize.x*imsize.y);
	
	
	float result = 0;
	int k_offset = (ksize / 2);
	for (int k = 0; k < ksize ; k++) {
		if (dir == X_DIR) {
			int x_input = k_offset - k + x_i;
			if (x_input > 0 && x_input < imsize.x) {
				result += input[x_input + imsize.x*y_i + imsize.x*imsize.y*z_i];
			}
		}
		else if (dir == Y_DIR) {
			int y_input = k_offset - k + y_i;
			if (y_input > 0 && y_input < imsize.y) {
				result += input[x_i + imsize.y*y_input + imsize.x*imsize.y*z_i];
			}
		}
		else if (dir == Z_DIR) {
			int z_input = k_offset - k + z_i;
			if (z_input > 0 && z_input < imsize.z) {
				result += input[x_i + imsize.y*y_i + imsize.x*imsize.y*z_input];
			}
		}
		
	}
	output[x_i + imsize.x*y_i + imsize.x*imsize.y*z_i] = result;
}


int main(int argc, char **argv)
{
	std::cout << sizeof(float) << std::endl;
	//Load Image
	
	cv::Mat image;
	image = cv::imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Original image", image);

	//Set up cubes
	int size = image.rows*image.cols * 256;
	float *cube_wi, *cube_w;
	int kernel_size = 7;
	float *kernel = (float*)malloc(kernel_size*sizeof(float));
	define_kernel(kernel, 3.0, kernel_size);


	float *dev_cube_wi, *dev_cube_w, *dev_cube_wi_out, *dev_cube_w_out, *dev_kernel;
	cube_wi = (float*)calloc(size, sizeof(float));
	cube_w = (float*)calloc(size, sizeof(float));
	//filling
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			for (int k = 0; k < 256; k++) {
				if (image.at<uchar>(i, j) == k) {
					cube_wi[i + image.rows*j + image.rows*image.cols*k] = k;
					cube_w[i + image.rows*j + image.rows*image.cols*k] = 1.0;
				}
			}
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

	/*convolution <<<1, size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, X_DIR);
	float * temp = dev_cube_wi;
	dev_cube_wi = dev_cube_wi_out;
	dev_cube_wi_out = temp;

	convolution <<<1, size >>>(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, X_DIR);
	temp = dev_cube_w;
	dev_cube_w = dev_cube_w_out;
	dev_cube_w_out = temp;
	convolution <<<1, size >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, Y_DIR);
	temp = dev_cube_wi;
	dev_cube_wi = dev_cube_wi_out;
	dev_cube_wi_out = temp;

	convolution <<<1, size >>>(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, Y_DIR);
	temp = dev_cube_w;
	dev_cube_w = dev_cube_w_out;
	dev_cube_w_out = temp;

	convolution << <1, size >> >(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, Z_DIR);
	temp = dev_cube_w;
	dev_cube_w = dev_cube_w_out;
	dev_cube_w_out = temp;
	convolution << <1, size >> >(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, Z_DIR);
	temp = dev_cube_w;
	dev_cube_w = dev_cube_w_out;
	dev_cube_w_out = temp;*/

	//allocate gpu image
	float * dev_image, *result_image;

	image.convertTo(image, CV_32F);
	int imsize = image.rows*image.cols;
	cudaMalloc(&dev_image, imsize*sizeof(float));
	cudaMemcpy(dev_image, image.ptr(), imsize*sizeof(float), cudaMemcpyHostToDevice);
	


	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);

	slicing << < grid, block >> > (dev_image , dev_cube_wi, dev_cube_w, image_dimensions);

	result_image = (float*)malloc(imsize*sizeof(float));
	cudaMemcpy(result_image, dev_image, imsize*sizeof(float), cudaMemcpyDeviceToHost);


	//cv::gpu::GpuMat dev_image(image.rows, image.cols, CV_8U, dev_image);

	cv::Mat output_imag(image.rows, image.cols, CV_32F, result_image);
	//dev_output_image.download(output_imag);

	//cudaFree(dev_cube_wi_out);
	//cudaFree(dev_cube_wi);

	//cudaFree(dev_cube_w_out);
	//cudaFree(dev_cube_w);
	//cudaFree(dev_image);

//	cudaFree(dev_kernel);
//	free(cube_w);
	//free(cube_wi);
	//free(kernel);

	cv::namedWindow("Filtered image", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Filtered image", output_imag);
	cv::imwrite("Result.jpg", output_imag * 256);
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
		output_kernel[i] = exp(-0.5*pow((size / 2 - i) / sigma, 2));
	}
}