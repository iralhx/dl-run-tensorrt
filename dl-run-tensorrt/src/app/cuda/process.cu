#include <driver_types.h>
#include <cuda_runtime.h>
//#include<crt/math_functions.h>
const int NUM_BOX_ELEMENT = 7;

void transpose_kernel_invoker(float *src,int num_bboxes,int num_elements,float *dst,cudaStream_t stream)
{
    int edge = num_bboxes*num_elements;
    int block =256;
    int gird = ceil(edge/(float)block);
    transpose_kernel<<<gird,block,0,stream>>>(src,num_bboxes,num_elements,dst,edge);
}

static __global__ void transpose_kernel(float *src,int num_bboxes, int num_elements,float *dst,int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position>=edge)
        return;
    dst[position]=src[position/num_elements+(position%num_elements)*num_bboxes];

}


static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects){  

    int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= num_bboxes) return;

    float* pitem     = predict + (4 + num_classes) * position;
    // float objectness = pitem[4];
    // if(objectness < confidence_threshold)
    //     return;

    float* class_confidence = pitem + 4;
    float confidence        = *class_confidence++;
    int label               = 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }

    // confidence *= objectness;
    if(confidence < confidence_threshold)
        return;
   
    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;
    // printf("index %d max_objects %d\n", index,max_objects);
    float cx         = pitem[0];
    float cy         = pitem[1];
    float width      = pitem[2];
    float height     = pitem[3];

    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;


    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects, cudaStream_t stream)
{
    int block = 256;
    int  grid =  ceil(num_bboxes / (float)block);
        
    decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, parray, max_objects);
}

static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
        
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 

void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
    int block = max_objects<256? max_objects:256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}