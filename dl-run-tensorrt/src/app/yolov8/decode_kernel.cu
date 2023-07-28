#include "decode_kernel.h"
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
//84*8400
//8400*84
__global__ void transpose_kernel(float *src,int dim1, int dim2,float *dst){

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= dim1 * dim2)
    {
        return;
    }
    int dx = position % dim2;
    int dy = position / dim2;
    int detIndex = dx * dim1 + dy;

    *(dst+ detIndex) = src[position];
}


void app::transposeDevice(float* src, int dim1, int dim2, float* dst)
{
    int total = dim1 * dim2;
    dim3 block = block_dims(total);
    dim3 grid = grid_dims(total);

    transpose_kernel << < grid, block >> > (src, dim1, dim2, dst);
}



static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}


const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
//前四个是坐标，后面全部是置信度
static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem = predict + (4 + num_classes) * position;

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
}

void app::decode_result(float* predict, int num_bboxes, int num_class, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects)
{
    dim3 block = block_dims(num_bboxes);
    dim3 grid = grid_dims(num_bboxes);
    decode_kernel << < grid, block >> > (predict, num_bboxes, num_class, confidence_threshold, invert_affine_matrix, parray, max_objects);
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


__global__ void warpaffine_kernel(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    float* d2s, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s[0];
    float m_y1 = d2s[1];
    float m_z1 = d2s[2];
    float m_x2 = d2s[3];
    float m_y2 = d2s[4];
    float m_z2 = d2s[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2;
    float c0, c1, c2;


    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else {
        int x_low = floorf(src_x);
        int y_low = floorf(src_y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;


        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
        uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
        uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
        uint8_t* v4 = src + y_high * src_line_size + x_high * 3;
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }
    // bgr -> rgb
    float temp = c2;
    c2 = c0;
    c0 = temp;

    //// normalization
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // rgbrgbrgb -> rrrgggbbb
    int area = dst_height * dst_width;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}


void app::preprocess_kernel_img(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    float* d2i, cudaStream_t stream) {
    int all = dst_width * dst_height;
    dim3 block = block_dims(all);
    dim3 grid = grid_dims(all);
    warpaffine_kernel << <grid, block, 0, stream >> > (
        src, src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2i, all);

}
