#ifndef SLICING_H
#define SLICING_H

#include "include_file.h"

__global__ void slicing(float *dev_image, const float*dev_cube_wi, const float*dev_cube_w, const dim3 imsize);

void callingSlicing(float* result_image, float* dev_image, const float *dev_cube_wi, const float *dev_cube_w, const dim3 imsize);

#endif