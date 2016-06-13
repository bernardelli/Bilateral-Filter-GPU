#ifndef CUBEFILLING_H
#define CUBEFULLING_H

#include "include_file.h"

__global__ void cubefilling(cv::Mat image, float *cube_test_wi, float *cube_test_w);

void callingCubefilling(cv::Mat image, float *cube_test_wi, float *cube_test_w);

#endif