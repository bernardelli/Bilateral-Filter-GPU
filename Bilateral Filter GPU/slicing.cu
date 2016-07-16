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
		//printf("value = %d, i = %d, j = %d\n", value, i, j);


		//Old try on performing trilinear interpolation
		/*float interpolate_wi0[2][2][2];
		float interpolate_w0[2][2][2];
		
		for( int ii = 0; ii <2; ii++)  {
			int i_idx = floorf(i / scale_xy) + ii;
			for (int jj = 0; jj <2; jj++)  { 
				int j_idx = floorf(j / scale_xy) + jj;
				for (int kk = 0; kk <2; k++)  { 
					int k_idx = floorf(k / scale_eps) + kk;
					int cube_idx = i_idx + dimensions_down.x*j_idx + dimensions_down.x*dimensions_down.y*k_idx;
					interpolate_wi0[ii][jj][kk] = dev_cube_wi[cube_idx];
					interpolate_w0[ii][jj][kk] = dev_cube_w[cube_idx];
				}
			}
		}
		float interpolate_wi1[2][2];
		float interpolate_w1[2][2];
		for( int ii = 0; ii <2; ii++)  {
			for (int jj = 0; jj <2; jj++)  { 
				float k_rest = (k/ scale_eps) - floorf(k/ scale_eps);
				interpolate_wi1[ii][jj] = (1.0-k_rest)*interpolate_wi0[ii][jj][0] + k_rest*interpolate_wi0[ii][jj][1];
				interpolate_w1[ii][jj] = (1.0-k_rest)*interpolate_w0[ii][jj][0] + k_rest*interpolate_w0[ii][jj][1];
			}
		}

		float interpolate_wi2[2];
		float interpolate_w2[2];
		for( int ii = 0; ii <2; ii++)  {
			float j_rest = (j/ scale_xy) - floorf(j/ scale_xy);

			interpolate_wi2[ii] = (1.0-j_rest)*interpolate_wi1[ii][0] + j_rest*interpolate_wi1[ii][1];
			interpolate_w2[ii] = (1.0-j_rest)*interpolate_w1[ii][0] + j_rest*interpolate_w1[ii][1];
			
		}
		float i_rest = (i/ scale_xy) - floorf(i/ scale_xy);*/


		//dev_image[j + imsize.y*i] = ((1.0-i_rest)*interpolate_wi2[0] + i_rest*interpolate_wi2[1])/((1.0-i_rest)*interpolate_w2[0] + i_rest*interpolate_w2[1]);
		dev_image[i + imsize.x*j] = tex3D(wi_tex, 0.5 + (float)i / (float)scale_xy, 0.5 + (float)j / (float)scale_xy, 0.5 + (float)k / (float)scale_eps) / tex3D(w_tex, 0.5 + (float)i / (float)scale_xy, 0.5 + (float)j / (float)scale_xy, 0.5 + (float)k / (float)scale_eps);
		
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
	wi_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	wi_tex.addressMode[0] = cudaAddressModeMirror; //cudaAddressModeClamp
	wi_tex.addressMode[1] = cudaAddressModeMirror;
	wi_tex.addressMode[2] = cudaAddressModeMirror;
	w_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	w_tex.addressMode[0] = cudaAddressModeMirror;
	w_tex.addressMode[1] = cudaAddressModeMirror;
	w_tex.addressMode[2] = cudaAddressModeMirror;
	
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on gettexref " << cudaGetErrorString(cudaStatus) << std::endl;
	}
		
	cudaStatus = cudaBindTextureToArray(wi_tex_ref, dev_cube_wi_array, &channelFloat);//, cudaChannelFormatKindFloat); 	
	cudaBindTextureToArray(w_tex_ref, dev_cube_w_array, &channelFloat);//, cudaChannelFormatKindFloat);
	
	if (cudaStatus != cudaSuccess) {
		std::cout << "error on bind text " << cudaGetErrorString(cudaStatus) << std::endl;
	}
	
	const dim3 block2(16, 16);


	const dim3 grid2(((imsize.x + block2.x - 1) / block2.x), ((imsize.y + block2.y - 1) / block2.y));
	
	slicing <<< grid2, block2 >>> (dev_image, imsize, scale_xy, scale_eps);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	slicing <<< grid2, block2 >>> (dev_image, dev_cube_wi, dev_cube_w, imsize, scale_xy, scale_eps, dimensions_down);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	
	cudaUnbindTexture(wi_tex);
	cudaUnbindTexture(w_tex);
	cudaFreeArray(dev_cube_wi_array);
	cudaFreeArray(dev_cube_w_array);
	return time;
}
