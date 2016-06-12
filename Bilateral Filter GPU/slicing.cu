#include "slicing.h"

__global__ void slicing( float *dev_image, const float *dev_cube_wi, const float *dev_cube_w, const dim3 imsize)
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

float* callingSlicing( cv::Mat image, float *dev_cube_wi, float *dev_cube_w)
{
	float *dev_image, *result_image; 
	image.convertTo(image, CV_32F);
	int imsize = image.rows*image.cols;
	cudaMalloc(&dev_image, imsize*sizeof(float));
	cudaMemcpy(dev_image, image.ptr(), imsize*sizeof(float), cudaMemcpyHostToDevice);
	
	//Specify a reasonable block size
	const dim3 block2(32, 32);

	//Calculate grid size to cover the whole image
	const dim3 grid2(((image.cols + block2.x - 1) / block2.x), ((image.rows + block2.y - 1) / block2.y));
	
	dim3  image_dimensions = dim3(image.rows, image.cols, 256);
	
	slicing <<< grid2, block2 >>> (dev_image , dev_cube_wi, dev_cube_w, image_dimensions);
	cudaDeviceSynchronize();

	result_image = (float*)malloc(imsize*sizeof(float));
	cudaMemcpy(result_image, dev_image, imsize*sizeof(float), cudaMemcpyDeviceToHost);
}