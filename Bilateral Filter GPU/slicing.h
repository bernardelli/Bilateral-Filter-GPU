#ifndef SLICING_H
#define SLICING_H

#include "include_file.h"

__global__ void slicing(float *dev_image, const float*dev_cube_wi, const float*dev_cube_w, const dim3 imsize, int scale_xy, int scale_eps, dim3 dimensions_down);

float callingSlicing(float* dev_image, const float *dev_cube_wi, const float *dev_cube_w, const dim3 imsize, int scale_xy, int scale_eps, dim3 dimensions_down);

#endif
