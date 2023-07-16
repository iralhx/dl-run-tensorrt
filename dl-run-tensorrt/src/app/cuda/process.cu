#include <driver_types.h>
#include <cuda_runtime.h>
//#include<crt/math_functions.h>
const int NUM_BOX_ELEMENT = 7;

__global__ void transpose_kernel(float *src,int num_bboxes, int num_elements,float *dst,int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
}
