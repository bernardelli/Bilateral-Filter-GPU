#ifndef SLICING_H
#define SLICING_H

__global__ void slicing(float *dev_image, const float*dev_cube_wi, const float*dev_cube_w, const dim3 imsize);

float* callingSlicing( cv::Mat image, float *dev_cube_wi, float *dev_cube_w);

#endif