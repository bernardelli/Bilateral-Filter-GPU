#ifndef CUBEFILLING_H
#define CUBEFULLING_H

#include "little_cuda_functions.h"
#include "include_file.h"

__global__ void cubefilling(cv::Mat image, float *cube_test_wi, float *cube_test_w);

void callingCubefilling(const float* image, float *cube_test_wi, float *cube_test_w, const dim3 size);

#endif