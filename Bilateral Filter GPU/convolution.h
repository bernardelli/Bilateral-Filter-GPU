#ifndef CONVOLUTION_H
#define CONVOLUTION_H

__global__ void convolution(float *output, const float *input, const float* kernel, const int ksize,  const dim3 imsize, const int dir);

void callingConvolution(cv::Mat image, float *dev_cube_wi_out, float *dev_cube_w_out, float *dev_cube_wi, int dev_kernel, int kernel_size);

void swap(int*& a, int*& b);

#endif