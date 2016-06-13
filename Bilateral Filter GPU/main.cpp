#include "include_file.h"


#include "slicing.h"
#include "convolution.h"
#include "little_cuda_functions.h"
#include "cubefilling.cuh"


int main(int argc, char **argv)
{
	
	/********************************************************************************
	*** initialization of variables                                               ***
	********************************************************************************/
	cv::Mat image;
	int size, kernel_size, image_size;
	float *kernel, *dev_cube_wi, *dev_cube_w, *dev_cube_wi_out, 
	      *dev_cube_w_out, *dev_kernel, *dev_image, *result_image;
	cudaError_t cudaStatus;


	
	
	/********************************************************************************
	*** printing compute capability of each device                                ***
	********************************************************************************/
	checkingDevices();
	
	
	/********************************************************************************
	*** loading image and display it on desktop                                   ***
	********************************************************************************/
	image = cv::imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file


	if (!image.data) {
		std::cerr << "Could not open or find \"lena.bmp\"." << std::endl;
		return 1;
	}
	
	image_size = image.rows*image.cols;
	size = image_size * 256;
	dim3 dimensions = dim3(image.rows, image.cols, 256);

	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original image", image);
	image.convertTo(image, CV_32F);
	
	/********************************************************************************
	*** define kernel                                                             ***
	********************************************************************************/
	
	kernel_size = 57;
	kernel = (float*)malloc(kernel_size*sizeof(float));
	define_kernel(kernel, 25.5, kernel_size);

	
	
	
	/********************************************************************************
	*** choose which GPU to run on, change this on a multi-GPU system             ***
	********************************************************************************/
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	
	
	/********************************************************************************
	*** allocate the space for cubes on gpu memory                                ***
	********************************************************************************/
	cudaStatus = allocateGpuMemory(&dev_cube_wi, size);
	cudaStatus = allocateGpuMemory(&dev_cube_w, size);
	cudaStatus = allocateGpuMemory(&dev_cube_wi_out, size);
	cudaStatus = allocateGpuMemory(&dev_cube_w_out, size);
	cudaStatus = allocateGpuMemory(&dev_kernel, kernel_size);
	cudaStatus = allocateGpuMemory(&dev_image, image_size);
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	

	

	
	/********************************************************************************
	*** copy cubes on gpu memory                                                  ***
	********************************************************************************/
	//cudaStatus = copyToGpuMem(dev_cube_wi,cube_wi, size);
	//cudaStatus = copyToGpuMem(dev_cube_w,cube_w, size);
	cudaStatus = copyToGpuMem(dev_kernel, kernel, kernel_size);
	cudaStatus = cudaMemcpy(dev_image, image.ptr(), image_size*sizeof(float), cudaMemcpyHostToDevice);////copyToGpuMem(dev_image,(float*) image.ptr(), size); //only works with raw function!
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	

	/********************************************************************************
	*** setting up the cubes and filling them                                     ***
	********************************************************************************/

	callingCubefilling(dev_image, dev_cube_wi, dev_cube_w, dimensions);

	
	/********************************************************************************
	*** start concolution on gpu                                                  ***
	********************************************************************************/
	callingConvolution(dev_cube_wi_out, dev_cube_w_out, dev_cube_wi, dev_cube_w, dev_kernel, kernel_size, dimensions);
	
	
	/********************************************************************************
	*** start slicing on gpu                                                      ***
	********************************************************************************/
	result_image = (float*)malloc(image_size*sizeof(float));
	callingSlicing(result_image, dev_image, dev_cube_wi, dev_cube_w, dimensions);
	cv::Mat output_imag(image.rows, image.cols, CV_32F, result_image);
	
	/********************************************************************************
	*** free every malloced space                                                 ***
	********************************************************************************/
	cudaFree(dev_cube_wi_out);
	cudaFree(dev_cube_wi);
	cudaFree(dev_cube_w_out);
	cudaFree(dev_cube_w);
	cudaFree(dev_kernel);
	cudaFree(dev_image);
	free(kernel);
	
	
	
	/********************************************************************************
	*** show filtered image and save image                                        ***
	********************************************************************************/
	
	cv::namedWindow("Filtered image", cv::WINDOW_AUTOSIZE);

	cv::imshow("Filtered image", output_imag/256);
	cv::imwrite("Result.bmp", output_imag);
	cv::waitKey(0);
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
