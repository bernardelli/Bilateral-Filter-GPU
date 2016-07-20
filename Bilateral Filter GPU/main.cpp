﻿#include "include_file.h"

#include "slicing.cuh"
#include "convolution_sep.cuh"
#include "little_cuda_functions.cuh"
#include "cubefilling.cuh"


int main(int argc, char **argv)
{
	
	
	/********************************************************************************
	*** initialization of variables                                               ***
	********************************************************************************/
	cv::Mat image, image_original;
	int size, image_size, image_size_down, size_down, device;
	float *kernel_eps, *kernel_xy, *dev_cube_wi, *dev_cube_w, *dev_cube_wi_out, 
		*dev_cube_w_out, *dev_kernel_xy, *dev_kernel_eps, *dev_image, *result_image;
	char filename[40];
	cudaError_t cudaStatus;
	cudaDeviceProp deviceProp;
	
	
	/********************************************************************************
	*** printing compute capability of each device                                ***
	********************************************************************************/
	checkingDevices();
	
	
	/********************************************************************************
	*** choose which GPU to run on                                                ***
	********************************************************************************/
	device = 0;
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(1);
	}
	
	
	/********************************************************************************
	*** making the output file                                                    ***
	********************************************************************************/
	cudaGetDeviceProperties(&deviceProp, device);
	sprintf(filename, "%s.txt", deviceProp.name);
	FILE* output_file = fopen(filename, "w");
	fprintf(output_file,"%s\nRepeating\tImage size\tscale_xy\tscale_eps\tkernel_xy_size\tkernel_eps_size\tcubefilling\tconvolution\tslicing\ttime to allocate\ttime copy memory\ttime to free\tkomplete time\n\n", deviceProp.name);
	

	/********************************************************************************
	*** loading image and display it on desktop                                   ***
	********************************************************************************/
	image_original = cv::imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	
	if (!image_original.data) {
		std::cerr << "Could not open or find \"lena.bmp\"." << std::endl;
		return 1;
	}
	

#ifndef  __linux__
	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original image", image);
#endif
	
	
	
	for (int repeating = 0; repeating < 10; repeating++){
		
		// resizing the image size
		for (float image_resizing = 0.1; image_resizing <= 1.09; image_resizing += 0.1){
			
			cv::resize(image_original, image, cv::Size(image_original.cols * image_resizing, image_original.rows * image_resizing), 0, 0, CV_INTER_LINEAR);
			image.convertTo(image, CV_32F);
			image_size = image.rows*image.cols;
			size = image_size * 256;
			
			// resizing scale_xy
			for (int scale_xy = 1; scale_xy < 50; scale_xy += 5){
				
				// resizing scale_eps
				for (int scale_eps = 1; scale_eps < 50; scale_eps += 5){
					
					// resizing kernel_xy_size
					for (int kernel_xy_size = 1; kernel_xy_size < 50; kernel_xy_size += 5){
						
						// resizing kernel_eps_size
						for (int kernel_eps_size = 1; kernel_eps_size < 50; kernel_eps_size += 5){
							
							// start measuring the complete time of the filtering
							cudaEvent_t start_kom, stop_kom;
							float time_kom;
							cudaEventCreate(&start_kom);
							cudaEventCreate(&stop_kom);
							cudaEventRecord(start_kom);
							
							image_size_down = (image.rows /scale_xy)*(image.cols / scale_xy);
							
							size_down = image_size_down*(256 / scale_eps);
							dim3 dimensions = dim3(image.rows, image.cols, 256);
							dim3 dimensions_down = dim3((image.rows / scale_xy), (image.cols / scale_xy), (256 / scale_eps));
							
							/********************************************************************************
							*** calculate sigma_xy and sigma_eps                                          ***
							********************************************************************************/
							float sigma_xy = 25.0f / scale_xy;
							kernel_xy = (float*)malloc(kernel_xy_size*sizeof(float));
							define_kernel(kernel_xy, sigma_xy, kernel_xy_size);
							
							float sigma_eps = 25.0f / scale_eps;
							kernel_eps = (float*)malloc(kernel_eps_size*sizeof(float));
							define_kernel(kernel_eps, sigma_eps, kernel_eps_size);
							
							
							
							/********************************************************************************
							*** allocate the space for cubes on gpu memory (with time measure)            ***
							********************************************************************************/
							
							cudaEvent_t start, stop;
							float time_allocate, time_gpumem1, time_gpumem2, time_free;
							cudaEventCreate(&start);
							cudaEventCreate(&stop);
							
							cudaEventRecord(start);
							
							cudaStatus = allocateGpuMemory(&dev_cube_wi, size_down);
							cudaStatus = allocateGpuMemory(&dev_cube_w, size_down);
							cudaStatus = allocateGpuMemory(&dev_cube_wi_out, size_down);
							cudaStatus = allocateGpuMemory(&dev_cube_w_out, size_down);
							cudaStatus = allocateGpuMemory(&dev_kernel_xy, kernel_xy_size);
							cudaStatus = allocateGpuMemory(&dev_kernel_eps, kernel_eps_size);
							cudaStatus = allocateGpuMemory(&dev_image, image_size);
							
							cudaEventRecord(stop);
							cudaEventSynchronize(stop);
							
							cudaEventElapsedTime(&time_allocate, start, stop);
							
							if (cudaStatus != cudaSuccess) {
								fprintf(stderr, "cudaMalloc failed!");
							}
							
							
							/********************************************************************************
							*** copy data on gpu memory (with time measure)                               ***
							********************************************************************************/
							cudaEventRecord(start);
							
							cudaStatus = copyToGpuMem(dev_kernel_xy, kernel_xy, kernel_xy_size);
							cudaStatus = copyToGpuMem(dev_kernel_eps, kernel_eps, kernel_eps_size);
							cudaStatus = cudaMemcpy(dev_image, image.ptr(), image_size*sizeof(float), cudaMemcpyHostToDevice);////copyToGpuMem(dev_image,(float*) image.ptr(), size); //only works with raw function!
							
							cudaEventRecord(stop);
							cudaEventSynchronize(stop);
							
							cudaEventElapsedTime(&time_gpumem1, start, stop);
							
							if (cudaStatus != cudaSuccess) {
								fprintf(stderr, "cudaMemcpy failed!");
							}
							

							/********************************************************************************
							*** setting up the cubes and filling them                                     ***
							********************************************************************************/
							float cubefilling_time = callingCubefilling(dev_image, dev_cube_wi, dev_cube_w, dimensions, scale_xy, scale_eps, dimensions_down);
							
							
							/********************************************************************************
							*** start concolution on gpu                                                  ***
							********************************************************************************/
							float convolution_time = callingConvolution_sep(dev_cube_wi_out, dev_cube_w_out, dev_cube_wi, dev_cube_w, dev_kernel_xy, kernel_xy_size, dev_kernel_eps, kernel_eps_size, dimensions_down, device);
							
							
							/********************************************************************************
							*** start slicing on gpu                                                      ***
							********************************************************************************/
							result_image = (float*)malloc(image_size*sizeof(float));
							float slicing_time = callingSlicing(dev_image, dev_cube_wi_out, dev_cube_w_out, dimensions,scale_xy, scale_eps, dimensions_down);
							
							
							/********************************************************************************
							*** write back of result of filtering from gpu to host                        ***
							********************************************************************************/
							cudaEventRecord(start);
							cudaMemcpy(result_image, dev_image, dimensions.x*dimensions.y*sizeof(float), cudaMemcpyDeviceToHost);
							cudaEventRecord(stop);
							cudaEventSynchronize(stop);
							
							cudaEventElapsedTime(&time_gpumem2, start, stop);
							
							cv::Mat output_imag(image.rows, image.cols, CV_32F, result_image);
							
							
							//std::cout << "Total time: " << cubefilling_time + convolution_time + slicing_time << " ms" << std::endl;
							
							
							/********************************************************************************
							*** free every malloced space                                                 ***
							********************************************************************************/
							cudaEventRecord(start);
							cudaFree(dev_cube_wi_out);
							cudaFree(dev_cube_wi);
							cudaFree(dev_cube_w_out);
							cudaFree(dev_cube_w);
							cudaFree(dev_kernel_xy);
							cudaFree(dev_kernel_eps);
							cudaFree(dev_image);
							free(kernel_xy);
							free(kernel_eps);
							cudaEventRecord(stop);
							cudaEventSynchronize(stop);
							
							cudaEventElapsedTime(&time_free, start, stop);
							
							
							/********************************************************************************
							*** show filtered image and save image                                        ***
							********************************************************************************/
							char result_image_[200];
							sprintf(result_image_, "/scratch-local/mapprakt4/result_images/Result_%d_%d_%d_%d_%d.bmp", image.rows, scale_xy, scale_eps, kernel_xy_size, kernel_eps_size);
							cv::imwrite(result_image_, output_imag);
							cudaEventRecord(stop_kom);
							cudaEventSynchronize(stop_kom);
							
							cudaEventElapsedTime(&time_kom, start_kom, stop_kom);
							
							
							/********************************************************************************
							*** printing the solutions in an output file                                  ***
							********************************************************************************/
							fprintf(output_file, "%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", 
								repeating, image.rows, scale_xy, scale_eps, kernel_xy_size, kernel_eps_size, cubefilling_time, convolution_time, slicing_time, time_allocate, time_gpumem1 + time_gpumem2, time_free, time_kom);
						}
					}
				}
			}
		}
	}
	
	fclose(output_file);
	
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
		output_kernel[i] = expf(-0.5*powf((size / 2.0f - i) / sigma, 2));
	}
}
