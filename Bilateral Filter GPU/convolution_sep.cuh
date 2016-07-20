#ifndef CONVOLUTION_SEP_CUH
#define CONVOLUTION_SEP_CUH

#define X_DIR 0 
#define Y_DIR 1 
#define EPS_DIR 2 
#include "include_file.h"


__global__ void convolution_sep(float *, const float *, const float*, const int, int dir);
float callingConvolution_sep(float *dev_cube_wi_out, float *dev_cube_w_out, float *dev_cube_wi, float *dev_cube_w, const float *dev_kernel_xy, int kernel_xy_size, const float *dev_kernel_eps, int kernel_eps_size, dim3  image_dimensions, int device);

void swap2(float** a, float** b);

#endif