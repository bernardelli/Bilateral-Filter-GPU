#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "include_file.h"

__global__ void convolution(float *, const float *, const float* , const int ,  const dim3 , const int );

void callingConvolution(float *dev_cube_wi_out, float *dev_cube_w_out, float *dev_cube_wi, float *dev_cube_w, const float *dev_kernel_xy, int kernel_xy_size, const float *dev_kernel_eps, int kernel_eps_size, dim3  image_dimensions);

void swap(float** a, float** b);

#endif