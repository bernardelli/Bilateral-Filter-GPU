#include <include_file.h>

int main(int argc, char **argv)
{
	
	// printing compute capability of each device
	checkingDevices();
/*	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d and concurrentKernels = %d.\n",
			device, deviceProp.major, deviceProp.minor, deviceProp.concurrentKernels);
	}
	cudaDeviceReset();
*/	
	
	//Load Image
	cv::Mat image;
	image = cv::imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	
	if (!image.data) {
		std::cout << "Could not open or find \"lena.bmp\"." << std::endl;
		return 1;
	}
	
	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Original image", image);

	//Set up cubes
	int size = image.rows*image.cols * 256;
	float *cube_wi, *cube_w;
	int kernel_size = 57;
	float *kernel = (float*)malloc(kernel_size*sizeof(float));
	define_kernel(kernel, 25.5, kernel_size);

	
	// setting up the cubes and filling them
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
	
	
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	
	cudaStatus = allocateGpuMemory(&dev_cube_wi, size);
/*	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cube_wi, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
*/	
	
	cudaStatus = allocateGpuMemory(&dev_cube_w, size);
/*	cudaStatus = cudaMalloc((void**)&dev_cube_w, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
*/	
	
	cudaStatus = allocateGpuMemory(&dev_cube_wi_out, size);
/*	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cube_wi_out, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
*/	
	
	cudaStatus = allocateGpuMemory(&dev_cube_w_out, size);
/*	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cube_w_out, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
*/
	
	cudaStatus = allocateGpuMemory(&dev_kernel, kernel_size);
/*	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_kernel, kernel_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
*/
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	
	cudaStatus = copyToGpuMem(dev_cube_wi,cube_wi, size);
/*	cudaStatus = cudaMemcpy(dev_cube_wi, cube_wi, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
*/
	
	cudaStatus = copyToGpuMem(dev_cube_w,cube_w, size);
/*	cudaStatus = cudaMemcpy(dev_cube_w, cube_w, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
*/
	
	cudaStatus = copyToGpuMem(dev_kernel,kernel, kernel_size);
//	cudaStatus = cudaMemcpy(dev_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	
	
	// block 64x64 and grid 128x128 or block 32 32 and grid 256 256
	//ITS WRONG BUT WORKS FOR NOT A LOT OF THREADS IDK
//	const dim3 block(32, 32); //threads per block 32 32

	//Calculate grid size to cover the whole image
	//int grid_n = ceil(sqrt(size) / (block.y));
//	int grin = 256;    //256 made it work with 231
//	const dim3 grid(grin, grin); //number of blocks
	//int block = 1024;
	//int grid = size / block;
	
	callingConvolution(image, dev_cube_wi_out, dev_cube_w_out, dev_cube_wi, dev_kernel, kernel_size);
	
/*	dim3  image_dimensions = dim3(image.rows, image.cols, 256);
	
	convolution <<< grid, block >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	swap(dev_cube_wi, dev_cube_wi_out);

	convolution <<< grid, block >>>(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, X_DIR);
	cudaDeviceSynchronize();
	swap(dev_cube_w, dev_cube_w_out);
		
	convolution <<< grid, block >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	swap(dev_cube_wi, dev_cube_wi_out);
	
	convolution <<< grid, block >>>(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, Y_DIR);
	cudaDeviceSynchronize();
	swap(dev_cube_w, dev_cube_w_out);
	
	convolution <<< grid, block >>>(dev_cube_wi_out, dev_cube_wi, dev_kernel, kernel_size, image_dimensions, Z_DIR);
	cudaDeviceSynchronize();
	swap(dev_cube_wi, dev_cube_wi_out);
	
	convolution <<< grid, block >>>(dev_cube_w_out, dev_cube_w, dev_kernel, kernel_size, image_dimensions, Z_DIR);
	cudaDeviceSynchronize();
	swap(dev_cube_w, dev_cube_w_out);
*/
	
	//allocate gpu image
	float *result_image;
	
	result_image = callingSlicing(image, dev_cube_wi, dev_cube_w);

	
/*	image.convertTo(image, CV_32F);
	int imsize = image.rows*image.cols;
	cudaMalloc(&dev_image, imsize*sizeof(float));
	cudaStatus = copyToGpuMem(dev_image,image.ptr(), imsize);
//	cudaMemcpy(dev_image, image.ptr(), imsize*sizeof(float), cudaMemcpyHostToDevice);
	


	//Specify a reasonable block size
	const dim3 block2(32, 32);

	//Calculate grid size to cover the whole image
	const dim3 grid2(((image.cols + block2.x - 1) / block2.x), ((image.rows + block2.y - 1) / block2.y));

	slicing <<< grid2, block2 >>> (dev_image , dev_cube_wi, dev_cube_w, image_dimensions);
	cudaDeviceSynchronize();

	result_image = (float*)malloc(imsize*sizeof(float));
	cudaMemcpy(result_image, dev_image, imsize*sizeof(float), cudaMemcpyDeviceToHost);

*/
	//cv::gpu::GpuMat dev_image(image.rows, image.cols, CV_8U, dev_image);

	cv::Mat output_imag(image.rows, image.cols, CV_32F, result_image);



	
	//std::cout << output_imag << std::endl;
	//dev_output_image.download(output_imag);
	
	
	// free every malloced space
	cudaFree(dev_cube_wi_out);
	cudaFree(dev_cube_wi);
	cudaFree(dev_cube_w_out);
	cudaFree(dev_cube_w);
	cudaFree(dev_image);
	cudaFree(dev_kernel);
	free(cube_w);
	free(cube_wi);
	free(kernel);
	
	
	// show filtered image and save image
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

void swap(int*& a, int*& b){
    int* c = a;
    a = b;
    b = c;
}