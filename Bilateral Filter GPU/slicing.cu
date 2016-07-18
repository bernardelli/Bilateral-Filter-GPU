#include "slicing.h"

texture<float, 3> wi_tex;
texture<float, 3> w_tex;

__global__ void slicing( float *dev_image, const dim3 imsize, int scale_xy, int scale_eps)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((i < imsize.x) && (j < imsize.y))
	{

		float k = dev_image[i + imsize.x*j];


		dev_image[i + imsize.x*j] = 256*tex3D(wi_tex, 0.5f + (float)i / (float)scale_xy, 0.5f + (float)j / (float)scale_xy, 0.5f + (float)k / (float)scale_eps) 
									/ tex3D(w_tex, 0.5f + (float)i / (float)scale_xy, 0.5f + (float)j / (float)scale_xy, 0.5f + (float)k / (float)scale_eps);
		
	}

}

/*__global__ void fill_arrays(cudaArray* dev_cube_wi_array, cudaArray* dev_cube_w_array, const float* dev_cube_wi, const float* dev_cube_w, const dim3 )
{

	const int ix = blockDim.x*blockIdx.x + threadIdx.x;
	const int iy = blockDim.y*blockIdx.y + threadIdx.y;
	const int iz = blockIdx.z;
	const int cube_idx = ix + iy*dimensions_down.x + iz*dimensions_down.x*dimensions_down.y;
	if(ix < dimensions_down.x &&  iy < dimensions_down.y && iy < dimensions_down.y )
	{
		dev_cube_wi_array = dev_cube_wi[cube_idx];
	}
}*/

float callingSlicing(float* dev_image, const float *dev_cube_wi, const float *dev_cube_w, const dim3 imsize, int scale_xy, int scale_eps, dim3 dimensions_down)
{
	int slicing_status = 0;
	const dim3 block2(16, 16);
	wi_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	wi_tex.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeClamp
	wi_tex.addressMode[1] = cudaAddressModeClamp;
	wi_tex.addressMode[2] = cudaAddressModeClamp;
	w_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	w_tex.addressMode[0] = cudaAddressModeClamp;
	w_tex.addressMode[1] = cudaAddressModeClamp;
	w_tex.addressMode[2] = cudaAddressModeClamp;
	cudaExtent extent1 = make_cudaExtent( dimensions_down.x*sizeof(float), dimensions_down.y, dimensions_down.z); 
	cudaExtent extent2 = make_cudaExtent( dimensions_down.x, dimensions_down.y, dimensions_down.z); 
	cudaArray *dev_cube_wi_array, *dev_cube_w_array;

	cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();
	cudaError_t cudaStatus = cudaMalloc3DArray(&dev_cube_wi_array, &channelFloat, extent2);
	cudaStatus = cudaMalloc3DArray(&dev_cube_w_array, &channelFloat, extent2);
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on malloc3darray " << cudaGetErrorString(cudaStatus) << std::endl;
	}

	cudaMemcpy3DParms copyParams1 = { 0 };
	copyParams1.srcPtr = make_cudaPitchedPtr((void*) dev_cube_wi,
		dimensions_down.x* sizeof(float), //https://devtalk.nvidia.com/default/topic/481806/copy-3d-data-from-host-to-device/
		dimensions_down.x, dimensions_down.y);
	copyParams1.dstArray = dev_cube_wi_array;
	copyParams1.extent = extent2;
	copyParams1.kind = cudaMemcpyDeviceToDevice;
	cudaStatus = cudaMemcpy3D(&copyParams1);

	if (cudaStatus != cudaSuccess) {
		std::cout << "error on copying to array1" << std::endl;
		slicing_status = 1;
	}
	cudaMemcpy3DParms copyParams2 = { 0 };
	copyParams2.srcPtr = make_cudaPitchedPtr((void*) dev_cube_w,
		dimensions_down.x * sizeof(float),
		dimensions_down.x, dimensions_down.y);
	copyParams2.dstArray = dev_cube_w_array;
	copyParams2.extent = extent2;
	copyParams2.kind = cudaMemcpyDeviceToDevice;
	cudaStatus = cudaMemcpy3D(&copyParams2);
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on copying to array2"<< std::endl;
		slicing_status = 1;
	}
	//cudaMemcpyToArray(dev_cube_wi_array, 0, 0, dev_cube_wi, dimensions_down.x*dimensions_down.y*dimensions_down.z, cudaMemcpyDeviceToDevice);
	//cudaMemcpyToArray(dev_cube_w_array, 0, 0, dev_cube_w, dimensions_down.x*dimensions_down.y*dimensions_down.z, cudaMemcpyDeviceToDevice);
	//fill_arrays<<grid,block>>(dev_cube_wi_array, dev_cube_w_array,dev_cube_wi, dev_cube_w,dimensions_down);

	//struct cudaChannelFormatDesc descr = cudaCreateChannelDesc((int)dimensions_down.x, (int)dimensions_down.y, (int)dimensions_down.z, cudaChannelFormatKindFloat);
	const struct textureReference * wi_tex_ref;
	const struct textureReference * w_tex_ref;
	
#if CUDA_VERSION < 5000 /* 5.0 */
	cudaStatus = cudaGetTextureReference(&wi_tex_ref, "wi_tex");
	cudaGetTextureReference(&w_tex_ref, "w_tex");
#else
	cudaStatus = cudaGetTextureReference(&wi_tex_ref, &wi_tex);
	cudaGetTextureReference(&w_tex_ref, &w_tex);
#endif

	
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on gettexref " << cudaGetErrorString(cudaStatus) << std::endl;
	}
		
	cudaStatus = cudaBindTextureToArray(wi_tex_ref, dev_cube_wi_array, &channelFloat);//, cudaChannelFormatKindFloat); 	
	cudaBindTextureToArray(w_tex_ref, dev_cube_w_array, &channelFloat);//, cudaChannelFormatKindFloat);
	
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on bind text " << cudaGetErrorString(cudaStatus) << std::endl;
	}
	

	const dim3 grid2(((imsize.x + block2.x - 1) / block2.x), ((imsize.y + block2.y - 1) / block2.y));
	
	cudaEvent_t start_1, stop_1;
        float time_1;
        cudaEventCreate(&start_1);
        cudaEventCreate(&stop_1);

        cudaEventRecord(start_1);

	slicing <<< grid2, block2 >>> (dev_image, imsize, scale_xy, scale_eps);
	
	cudaEventRecord(stop_1);
        cudaEventSynchronize(stop_1);

        cudaEventElapsedTime(&time_1, start_1, stop_1);


	cudaEvent_t start_2, stop_2;
	float time_2;
	cudaEventCreate(&start_2);
	cudaEventCreate(&stop_2);
	
	cudaEventRecord(start_2);
	slicing <<< grid2, block2 >>> (dev_image, imsize, scale_xy, scale_eps);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_2);
	cudaEventSynchronize(stop_2);
	
	cudaEventElapsedTime(&time_2, start_2, stop_2);
	
	cudaUnbindTexture(wi_tex);
	cudaUnbindTexture(w_tex);
	cudaFreeArray(dev_cube_wi_array);
	cudaFreeArray(dev_cube_w_array);
	float time = time_1 + time_2;
	return time;
}
