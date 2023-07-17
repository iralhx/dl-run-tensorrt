#include "decode_kernel.h"
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
//1*84*8400
//1*8400*84
__global__ void transpose_kernel(float *src,int num_bboxes, int num_class,float *dst){

    int weidthIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int heightIndex = blockDim.y * blockIdx.y + threadIdx.y;
    if (weidthIndex>=num_bboxes||heightIndex>=(num_class+4))
    {
        return;
    }

    int globalIndex = heightIndex * num_bboxes+weidthIndex;
    int detIndex = weidthIndex * (num_class + 4) + heightIndex;
    *(dst+ detIndex) = src[globalIndex];
}


void app::transposeDevice(float* src, int num_bboxes, int num_class, float* dst)
{
    dim3 grid_size(ceil(num_bboxes  / BLOCK_SIZE),
        ceil((num_class+4)/ BLOCK_SIZE));
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    transpose_kernel << < grid_size, block_size >> > (src, num_bboxes, num_class, dst);
}





const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
//前四个是坐标，后面全部是置信度
static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects) {

    int one_block_all = blockDim.x * blockDim.y;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int grid_offset = one_block_all * gridDim.x * blockIdx.y + one_block_all * blockIdx.x;
    int index = grid_offset + tid;

    if (index >= num_bboxes) {
        return;
    }

    float* pitem = predict + (4 + num_classes) * index*sizeof(float);

    float* class_confidence = pitem + 4;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    if (confidence < confidence_threshold)
        return;

    int indexResult = atomicAdd(parray, 1);
    if (indexResult >= max_objects)
        return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    printf("Source,cx:%f,cy:%f,width:%f,height:%f\n",
        cx, cy, width, height);

    float* pout_item = parray + 1 + indexResult * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
    printf("这是第:%d,left:%f,top:%f,right:%f,bottom:%f,confidence:%f,label:%f\n", 
        indexResult , left, top, right,bottom,confidence,label);

};



void app::decode_result(float* predict, int num_bboxes, int num_class, float confidence_threshold, float* parray, int max_objects)
{
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size( ceil((num_class + 4) / BLOCK_SIZE),
        ceil(num_bboxes / BLOCK_SIZE));
    decode_kernel << < grid_size, block_size >> > (predict, num_bboxes, num_class, confidence_threshold, parray, max_objects);
}



static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom
) {

    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold) {

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]
            );

            if (iou > threshold) {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

void app::nms_kernel_invoker(float* parray, float nms_threshold, int max_objects) {

    dim3 block = block_dims (max_objects);
    dim3 grid = grid_dims(max_objects) ;
    nms_kernel << <grid, block >> > (parray, max_objects, nms_threshold);
}


