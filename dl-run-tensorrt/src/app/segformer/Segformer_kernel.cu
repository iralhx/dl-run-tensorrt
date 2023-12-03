#include "Segformer_kernel.h"
#include <opencv2/opencv.hpp>


static __global__ void process_imgage_kernel(uint8_t* input, float* output, int allPixel)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int blockNum = blockDim.x * blockDim.y;
    int block_offset = blockNum * blockIdx.x;
    int num_a_row = blockNum * gridDim.x;
    int row_offset = num_a_row * blockIdx.y;
    int position = tid + block_offset + row_offset;

    if (position >= allPixel) 
    {
        printf("input  CUOWU \n");

        return;
    }
    int index = position * 3;
    output[index] = input[index] / 255.0f;
    output[index + 1] = input[index + 1] / 255.0f;
    output[index + 2] = input[index + 2] / 255.0f;
    //printf("input %d  ,%d  , %d\n", p1, *p2, *p3);
    //printf("output %f  ,%f  , %f\n", output[index], output[index+1], output[index+2]);
    //uint8_t* p1 = input;  // 指向输入数组的第一个元素
    //printf("First channel value in the input: %d\n", position);
}



void app::process_imgage(uint8_t* input, float* output, int height, int width, cudaStream_t stream)
{

    // mask_weights is mask_dim(32 element) gpu pointer
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    dim3 block(32, 32);

    process_imgage_kernel << <grid, block, 0, stream >> > (input, output,height* width);
}




