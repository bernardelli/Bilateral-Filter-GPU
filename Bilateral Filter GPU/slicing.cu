/*
Slicing:

Perform slicing and nonlinearity. Uses texture unities.
*/

#include "slicing.cuh"

texture<float, 3> wi_tex;
texture<float, 3> w_tex;

__global__ void slicing( float *dev_image, const dim3 imsize, int scale_xy, int scale_eps)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < imsize.x) && (j < imsize.y))
	{

		float k = dev_image[i + imsize.x*j];

		float x = 0.5f + (float)i / (float)scale_xy;
		float y = 0.5f + (float)j / (float)scale_xy;
		float z = 0.5f + (float)k / (float)scale_eps;

		dev_image[i + imsize.x*j] = tex3D(wi_tex, x, y, z) / tex3D(w_tex, x, y, z);
		
	}

}



float callingSlicing(float* dev_image, const float *dev_cube_wi, const float *dev_cube_w, const dim3 imsize, int scale_xy, int scale_eps, dim3 dimensions_down)
{


	int slicing_status = 0;
	cudaError_t cudaStatus;


	/****************************************************************************************************************
	***	Set Texture proprierties																				*****
	*****************************************************************************************************************/
	wi_tex.filterMode = cudaFilterModeLinear;
	wi_tex.addressMode[0] = cudaAddressModeClamp; 
	wi_tex.addressMode[1] = cudaAddressModeClamp;
	wi_tex.addressMode[2] = cudaAddressModeClamp;
	w_tex.filterMode = cudaFilterModeLinear;      
	w_tex.addressMode[0] = cudaAddressModeClamp;
	w_tex.addressMode[1] = cudaAddressModeClamp;
	w_tex.addressMode[2] = cudaAddressModeClamp;

	/****************************************************************************************************************
	***	Create 3D Arrays																						*****
	*****************************************************************************************************************/

	cudaArray *dev_cube_wi_array, *dev_cube_w_array;
	cudaExtent extent = make_cudaExtent( dimensions_down.x, dimensions_down.y, dimensions_down.z); 
	cudaChannelFormatDesc channelFloat = cudaCreateChannelDesc<float>();

	cudaStatus = cudaMalloc3DArray(&dev_cube_wi_array, &channelFloat, extent);
	cudaStatus = cudaMalloc3DArray(&dev_cube_w_array, &channelFloat, extent);

	if (cudaStatus != cudaSuccess) {
		std::cout << "error on malloc3darray " << cudaGetErrorString(cudaStatus) << std::endl;
	}

	/*Copy arrays*/

	cudaMemcpy3DParms copyParams1 = { 0 };
	copyParams1.srcPtr = make_cudaPitchedPtr((void*) dev_cube_wi,
		dimensions_down.x* sizeof(float), //https://devtalk.nvidia.com/default/topic/481806/copy-3d-data-from-host-to-device/
		dimensions_down.x, dimensions_down.y);
	copyParams1.dstArray = dev_cube_wi_array;
	copyParams1.extent = extent;
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
	copyParams2.extent = extent;
	copyParams2.kind = cudaMemcpyDeviceToDevice;
	cudaStatus = cudaMemcpy3D(&copyParams2);
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on copying to array2"<< std::endl;
		slicing_status = 1;
	}

	/*Get texture references*/
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
	
	/*Bind textures to array*/
	cudaStatus = cudaBindTextureToArray(wi_tex_ref, dev_cube_wi_array, &channelFloat);
	cudaStatus = cudaBindTextureToArray(w_tex_ref, dev_cube_w_array, &channelFloat);
	
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on bind text " << cudaGetErrorString(cudaStatus) << std::endl;
	}
	
	/****************************************************************************************************************
	***	Actual Slicing kernel																					*****
	*****************************************************************************************************************/

	const dim3 block(16, 16);
	const dim3 grid(((imsize.x + block.x - 1) / block.x), ((imsize.y + block.y - 1) / block.y));
	
	cudaEvent_t start_1, stop_1;
    float time_1;
    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);

    cudaEventRecord(start_1);

	slicing <<< grid, block >>> (dev_image, imsize, scale_xy, scale_eps);
	
	cudaEventRecord(stop_1);
    cudaEventSynchronize(stop_1);

    cudaEventElapsedTime(&time_1, start_1, stop_1);
	float time = time_1;

	/*clean*/
	cudaUnbindTexture(wi_tex_ref);
	cudaUnbindTexture(w_tex_ref);
	cudaFreeArray(dev_cube_wi_array);
	cudaFreeArray(dev_cube_w_array);
	return time;
}
