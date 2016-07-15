#ifndef CUBEFILLING_H
#define CUBEFULLING_H

#include "little_cuda_functions.h"
#include "include_file.h"

__global__ void cubefilling(const float* image, float *cube_test_wi, float *cube_test_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down);

void callingCubefilling(const float* image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down);

#endif
