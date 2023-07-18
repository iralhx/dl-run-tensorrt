#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include<math.h>
#include<stdio.h>

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

#define GPU_BLOCK_THREADS 512
#define BLOCK_SIZE 32


dim3 grid_dims(int numJobs);

dim3 block_dims(int numJobs);