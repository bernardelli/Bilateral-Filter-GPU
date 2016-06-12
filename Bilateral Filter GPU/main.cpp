#include "include_file.h"


#include "slicing.h"
#include "convolution.h"
#include "little_cuda_functions.h"


int main(int argc, char **argv)
{
	
	/********************************************************************************
	*** initialization of variables                                               ***
	********************************************************************************/
	cv::Mat image;
	int size, kernel_size;
	float *cube_wi, *cube_w, *kernel, *dev_cube_wi, *dev_cube_w, *dev_cube_wi_out, 
	      *dev_cube_w_out, *dev_kernel, *result_image;
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
	
	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original image", image);
	
	
	/********************************************************************************
	*** define kernel                                                             ***
	********************************************************************************/
	size = image.rows*image.cols * 256;
	kernel_size = 57;
	kernel = (float*)malloc(kernel_size*sizeof(float));
	define_kernel(kernel, 25.5, kernel_size);

	
	/********************************************************************************
	*** setting up the cubes and filling them                                     ***
	********************************************************************************/
	cube_wi = (float*)calloc(size, sizeof(float));
	cube_w = (float*)calloc(size, sizeof(float));
	
	// TODO: make filling with gpu
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
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	
	
	/********************************************************************************
	*** copy cubes on gpu memory                                                  ***
	********************************************************************************/
	cudaStatus = copyToGpuMem(dev_cube_wi,cube_wi, size);
	cudaStatus = copyToGpuMem(dev_cube_w,cube_w, size);
	cudaStatus = copyToGpuMem(dev_kernel,kernel, kernel_size);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	
	
	/********************************************************************************
	*** start concolution on gpu                                                  ***
	********************************************************************************/
	callingConvolution(image, dev_cube_wi_out, dev_cube_w_out, dev_cube_wi, dev_cube_w, dev_kernel, kernel_size);
	
	
	/********************************************************************************
	*** start slicing on gpu                                                      ***
	********************************************************************************/
	result_image = callingSlicing(image, dev_cube_wi, dev_cube_w);
	
	
	/********************************************************************************
	*** free every malloced space                                                 ***
	********************************************************************************/
	cudaFree(dev_cube_wi_out);
	cudaFree(dev_cube_wi);
	cudaFree(dev_cube_w_out);
	cudaFree(dev_cube_w);
	cudaFree(dev_kernel);
	free(cube_w);
	free(cube_wi);
	free(kernel);
	
	
	/********************************************************************************
	*** show filtered image and save image                                        ***
	********************************************************************************/
	cv::Mat output_imag(image.rows, image.cols, CV_32F, result_image);
	cv::namedWindow("Filtered image", cv::WINDOW_AUTOSIZE);

	cv::imshow("Filtered image", output_imag/256);
	cv::imwrite("Result.bmp", output_imag);
	cv::waitKey(0);
	
	
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
