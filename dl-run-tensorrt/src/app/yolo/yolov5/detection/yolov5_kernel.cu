#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include"yolov5_kernel.h"


static __global__ void v5_decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem = predict + (5 + num_classes) * position;

    //整体置信度
    float obj_confidence = pitem[4];
    if (obj_confidence < confidence_threshold)
        return;

    float* class_confidence = pitem + 5;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    confidence *= obj_confidence;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;
    float cx = pitem[0];
    float cy = pitem[1];
    float width = pitem[2];
    float height = pitem[3];

    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);


    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

void app::v5_decode_result(float* predict, int num_bboxes, int num_class, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects)
{
    dim3 block = block_dims(num_bboxes);
    dim3 grid = grid_dims(num_bboxes);
    v5_decode_kernel << < grid, block >> > (predict, num_bboxes, num_class, confidence_threshold, invert_affine_matrix, parray, max_objects);
}