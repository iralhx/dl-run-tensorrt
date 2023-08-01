#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include"yolov8seg_kernel.h"

//前四个是坐标，后面全部是置信度,在后面就是掩码
static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, int num_mask,
    float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem = predict + (4 + num_classes+ num_mask) * position;

    float* class_confidence = pitem + 4;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    // confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;
    // printf("index %d max_objects %d\n", index,max_objects);
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
    *pout_item++ = position;//这里是mask的位置
}

void app::decode_seg_result(float* predict, int num_bboxes, int num_class,int sun_mask, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects)
{
    dim3 block = block_dims(num_bboxes);
    dim3 grid = grid_dims(num_bboxes);
    decode_kernel << < grid, block >> > (predict, num_bboxes, num_class, sun_mask,confidence_threshold, invert_affine_matrix, parray, max_objects);
}


static __global__ void decode_single_mask_kernel(int left, int top, float* mask_weights,
    float* mask_predict, int mask_width,
    int mask_height, unsigned char* mask_out,
    int mask_dim, int out_width, int out_height) {
    // mask_predict to mask_out
    // mask_weights @ mask_predict
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= out_width || dy >= out_height) return;

    int sx = left + dx;
    int sy = top + dy;
    if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
        mask_out[dy * out_width + dx] = 0;
        return;
    }

    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ++ic) {
        float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }

    float alpha = 1.0f / (1.0f + exp(-cumprod));
    mask_out[dy * out_width + dx] = alpha * 255;
}

void app::decode_single_mask(float left, float top, float* mask_weights, float* mask_predict,
    int mask_width, int mask_height, unsigned char* mask_out,
    int mask_dim, int out_width, int out_height, cudaStream_t stream) {
    // mask_weights is mask_dim(32 element) gpu pointer
    dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
    dim3 block(32, 32);

    decode_single_mask_kernel<<<grid, block, 0, stream >>>(
        left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width,
        out_height);
}
