#include "cubefilling.cuh"


/*
__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x ;
	c[index] = a[index] + b[index];
}
*/

__global__ void cubefilling(const float* image, float *cube_test_wi, float *cube_test_w, const dim3 image_size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < image_size.x && j < image_size.y) {
		unsigned int k=0;
//		k = image.at<uchar>(i, j);

		cube_test_wi[i + image.rows*j + image.rows*image.cols*k] = ((float)k);
		//if (k != 0)
		//	printf("cube_wi[%d + image.rows*%d + image.rows*image.cols*k] = %f\n", i, j, cube_test_wi[i + image.rows*j + image.rows*image.cols*k]);
		//std::cout << k << std::endl;
		cube_test_w[i + image.rows*j + image.rows*image.cols*k] = 1.0;
	}

//	unsigned int k = image.at<uchar>(threadIdx.x, threadIdx.y);

//	int index = threadIdx.x + image.rows * threadIdx.y + image.rows * image.cols * k;
}

void callingCubefilling(cv::Mat image, float *cube_test_wi, float *cube_test_w)
{
	dim3 dimBlock(16, 16);
	dim3 dimGrid((image.rows + dimBlock.x - 1) / dimBlock.x,
		(image.cols + dimBlock.y - 1) / dimBlock.y);

	int size = image.rows*image.cols;

	float* dev_image = allocateGpuMemory(size);

	float* image_ptr = image.ptr<float>();
	dim3 image_size = dim3(image.rows, image.cols);

	copyToGpuMem(dev_image, image_ptr, size);
	cubefilling <<< dimGrid, dimBlock >>>(dev_image, cube_test_w, cube_test_w, image_size);


}

