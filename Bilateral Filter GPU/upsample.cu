/*#include "upsample.cuh"

__global__ void upsample(float * dev_cube_wi_uplsampled, float *dev_cube_w_uplsampled, float * dev_cube_wi, float * dev_cube_w, const dim3 imsize, int scale_xy, int scale_eps, dim3 dimensions_down)
{

	tex3D(textureRef, float x, float y, float z)

}

void upsample(float * dev_cube_wi_uplsampled, float *dev_cube_w_uplsampled, float * dev_cube_wi, float * dev_cube_w , const dim3 dimensions, int scale_xy, int scale_eps, const dim3 dimensions_down)
{
	texture<float, 3> wi_tex;
	texture<float, 3> w_tex;

	texture<float, 2, cudaReadModeElementType> textureRef;
	cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc<float>();
	cudaBindTexture3D(0, wi_tex, dev_cube_wi, &channelDesc, dimensions.x, dimensions.y, dimensions.z);

//cudaAddressModeClamp
	const dim3 block2(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid2(((imsize.x + block2.x - 1) / block2.x), ((imsize.y + block2.y - 1) / block2.y));
	

	
	slicing <<< grid2, block2 >>> (dev_image, dev_cube_wi, dev_cube_w, imsize, scale_xy, scale_eps, dimensions_down);
	cudaDeviceSynchronize();

	
	cudaMemcpy(result_image, dev_image, imsize.x*imsize.y*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaUnbindTexture(wi_tex);
	cudaUnbindTexture(w_tex);
}*/
