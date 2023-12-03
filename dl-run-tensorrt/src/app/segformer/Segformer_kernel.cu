#include "Segformer_kernel.h"
#include <opencv2/opencv.hpp>


static __global__ void process_image_kernel(uint8_t* input, float* output, int allPixel)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int blockNum = blockDim.x * blockDim.y;
    int block_offset = blockNum * blockIdx.x;
    int num_a_row = blockNum * gridDim.x;
    int row_offset = num_a_row * blockIdx.y;
    int position = tid + block_offset + row_offset;

    if (position >= allPixel) 
    {
        return;
    }
    //RGB -> RRRRRRRRR;GGGGGGGG;BBBBBBBB
    int inputIndex = position * 3;
    output[position] = input[inputIndex] / 255.0f;
    output[position + allPixel] = input[inputIndex + 1] / 255.0f;
    output[position + allPixel * 2] = input[inputIndex + 2] / 255.0f;
    //printf("input %d  ,%d  , %d\n", p1, *p2, *p3);
    //printf("output %f  ,%f  , %f\n", output[index], output[index+1], output[index+2]);
    //uint8_t* p1 = input;  // 指向输入数组的第一个元素
    //printf("First channel value in the input: %d\n", position);
}



void app::process_image(uint8_t* input, float* output, int height, int width, cudaStream_t stream)
{

    // mask_weights is mask_dim(32 element) gpu pointer
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    dim3 block(32, 32);

    process_image_kernel << <grid, block, 0, stream >> > (input, output,height* width);
}

static __global__ void post_process_result_kernel(uint32_t* input, uint8_t* output, int count) 
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position>count)
    {
        return;
    }
    output[position] = input[position];
}
void app::post_process_result(uint32_t* input, uint8_t* output, int count, cudaStream_t stream)
{
    dim3 block = block_dims(count);
    dim3 grid = grid_dims(count);
    post_process_result_kernel << <grid, block, 0, stream >> > (input, output, count);
}
