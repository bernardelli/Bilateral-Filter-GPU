#ifndef INCLUDE_FILE_H
#define INCLUDE_FILE_H

#ifdef __INTELLLISENSE__
void __syncthreads();
#endif


#include <math.h>
#include <iostream>

// Shared Library Test Functions
//#include <helper_functions.h>  // CUDA SDK Helper functions
  
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>


#define X_DIR 0
#define Y_DIR 1
#define Z_DIR 2

void define_kernel(float* output_kernel, float sigma, int size);

#endif