#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include<math.h>

#define GPU_BLOCK_THREADS 512
#define BLOCK_SIZE 32


dim3 grid_dims(int numJobs);

dim3 block_dims(int numJobs);