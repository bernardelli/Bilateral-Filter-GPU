#include "include_file.h"


#include "slicing.h"
//#include "convolution.h"
#include "convolution_shared.h"
#include "little_cuda_functions.h"
#include "cubefilling.cuh"


int main(int argc, char **argv)
{
	
	
	/********************************************************************************
	*** initialization of variables                                               ***
	********************************************************************************/
	cv::Mat image;
	int size, kernel_eps_size, kernel_xy_size, image_size, image_size_down, size_down;
	float *kernel_eps, *kernel_xy, *dev_cube_wi, *dev_cube_w, *dev_cube_wi_out, 
		*dev_cube_w_out, *dev_kernel_xy, *dev_kernel_eps, *dev_image, *result_image;//,dev_cube_wi_uplsampled, dev_cube_w_uplsampled;
	cudaError_t cudaStatus;


	
	
	/********************************************************************************
	*** printing compute capability of each device                                ***
	********************************************************************************/
	checkingDevices();

	
	/********************************************************************************
	*** define scalling                                               ***
	********************************************************************************/
	int scale_xy = 5;
	int scale_eps = 5;

	
	/********************************************************************************
	*** define kernel                                                             ***
	********************************************************************************/
	float sigma_xy = 16.0 / scale_xy;
	kernel_xy_size = 16;
	kernel_xy = (float*)malloc(kernel_xy_size*sizeof(float));
	define_kernel(kernel_xy, sigma_xy, kernel_xy_size);

	float sigma_eps = 25.0 / scale_eps;
	kernel_eps_size = 23;
	kernel_eps = (float*)malloc(kernel_eps_size*sizeof(float));
	define_kernel(kernel_eps, sigma_eps, kernel_eps_size);
	/*Article:
	A consistent approximation is a sampling rate proportional to the Gaussian bandwidth (i.e. ss
	ss/sigmas ≈ sr /sigmar) to achieve similar accuracy on the whole SxR domain.*/

	/********************************************************************************
	*** loading image and display it on desktop                                   ***
	********************************************************************************/
	image = cv::imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file


	if (!image.data) {
		std::cerr << "Could not open or find \"lena.bmp\"." << std::endl;
		return 1;
	}
	

	//copyMakeBorder(image, image, sigma_xy / 2, sigma_xy / 2, sigma_xy / 2, sigma_xy / 2, IPL_BORDER_CONSTANT, 0);
#ifndef  __linux__
	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original image", image);
#endif
	image.convertTo(image, CV_32F);
	image_size = image.rows*image.cols;
	size = image_size * 256;
	image_size_down = ceil((float)image.rows / (float)scale_xy)*ceil((float)image.cols / (float)scale_xy);

	size_down = image_size_down*ceil((float)256/scale_eps);
	dim3 dimensions = dim3(image.rows, image.cols, 256);
	dim3 dimensions_down = dim3(ceil((float)image.rows / (float)scale_xy), ceil((float)image.cols / (float)scale_xy), ceil((float)256 / (float)scale_eps));

	
	
	
	
	/********************************************************************************
	*** choose which GPU to run on, change this on a multi-GPU system             ***
	********************************************************************************/
	int device = 0;
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	
	
	/********************************************************************************
	*** allocate the space for cubes on gpu memory                                ***
	********************************************************************************/
	cudaStatus = allocateGpuMemory(&dev_cube_wi, size_down);
	cudaStatus = allocateGpuMemory(&dev_cube_w, size_down);
	//cudaStatus = allocateGpuMemory(&dev_cube_wi_uplsampled, size);
	//cudaStatus = allocateGpuMemory(&dev_cube_w_uplsampled, size);
	cudaStatus = allocateGpuMemory(&dev_cube_wi_out, size_down);
	cudaStatus = allocateGpuMemory(&dev_cube_w_out, size_down);
	cudaStatus = allocateGpuMemory(&dev_kernel_xy, kernel_xy_size);
	cudaStatus = allocateGpuMemory(&dev_kernel_eps, kernel_eps_size);
	cudaStatus = allocateGpuMemory(&dev_image, image_size);
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	

	

	
	/********************************************************************************
	*** copy data on gpu memory                                                  ***
	********************************************************************************/
	cudaStatus = copyToGpuMem(dev_kernel_xy, kernel_xy, kernel_xy_size);
	cudaStatus = copyToGpuMem(dev_kernel_eps, kernel_eps, kernel_eps_size);
	cudaStatus = cudaMemcpy(dev_image, image.ptr(), image_size*sizeof(float), cudaMemcpyHostToDevice);////copyToGpuMem(dev_image,(float*) image.ptr(), size); //only works with raw function!
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	

	/********************************************************************************
	*** setting up the cubes and filling them                                     ***
	********************************************************************************/
	//maybe use cudaPitchedPtr for cubes
	float cubefilling_time = callingCubefilling(dev_image, dev_cube_wi, dev_cube_w, dimensions, scale_xy, scale_eps, dimensions_down);
	std::cout << "Filling ok with time = " << cubefilling_time << " ms" << std::endl;
	
	/********************************************************************************
	*** start concolution on gpu                                                  ***
	********************************************************************************/
	float convolution_time = callingConvolution_shared(dev_cube_wi_out, dev_cube_w_out, dev_cube_wi, dev_cube_w, dev_kernel_xy, kernel_xy_size, dev_kernel_eps, kernel_eps_size, dimensions_down, device);
    std::cout << "Convolution ok with time = " << convolution_time << " ms" << std::endl;

	/********************************************************************************
	*** start slicing on gpu                                                      ***
	********************************************************************************/
	
	result_image = (float*)malloc(image_size*sizeof(float));
	float slicing_time = callingSlicing(dev_image, dev_cube_wi_out, dev_cube_w_out, dimensions,scale_xy, scale_eps, dimensions_down);

	cudaMemcpy(result_image, dev_image, dimensions.x*dimensions.y*sizeof(float), cudaMemcpyDeviceToHost);
	cv::Mat output_imag(image.rows, image.cols, CV_32F, result_image);
	std::cout << "Slicing ok with time = " << slicing_time << " ms" << std::endl;
	
	/********************************************************************************
	*** free every malloced space                                                 ***
	********************************************************************************/
	cudaFree(dev_cube_wi_out);
	cudaFree(dev_cube_wi);
	cudaFree(dev_cube_w_out);
	cudaFree(dev_cube_w);
	//cudaFree(dev_cube_wi_upsampled);
	//cudaFree(dev_cube_w_upsampled);
	cudaFree(dev_kernel_xy);
	cudaFree(dev_kernel_eps);
	cudaFree(dev_image);
	free(kernel_xy);
	free(kernel_eps);
	
	
	
	/********************************************************************************
	*** show filtered image and save image                                        ***
	********************************************************************************/
	cv::imwrite("Result.bmp", output_imag);
#ifndef  __linux__
	cv::namedWindow("Filtered image", cv::WINDOW_AUTOSIZE);

	cv::imshow("Filtered image", output_imag/256);

	cv::waitKey(0);
#endif
	free(result_image); //needs to be freed after using output_imag
	
	/********************************************************************************
	*** cudaDeviceReset must be called before exiting in order for profiling and  ***
    *** tracing tools such as Nsight and Visual Profiler to show complete traces. ***
	********************************************************************************/
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
