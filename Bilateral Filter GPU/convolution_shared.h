#ifndef CONVOLUTION_SHARED_H
#define CONVOLUTION_SHARED_H

#define BLOCK_DIM 32

#include "include_file.h"

__global__ void convolution_shared_row(float *, const float *, const float*, const int, const dim3);
__global__ void convolution_shared_col(float *, const float *, const float*, const int, const dim3);
__global__ void convolution_shared_eps(float *, const float *, const float*, const int, const dim3);
float callingConvolution_shared(float *dev_cube_wi_out, float *dev_cube_w_out, float *dev_cube_wi, float *dev_cube_w, const float *dev_kernel_xy, int kernel_xy_size, const float *dev_kernel_eps, int kernel_eps_size, dim3  image_dimensions, int device);

void swap2(float** a, float** b);

#endif