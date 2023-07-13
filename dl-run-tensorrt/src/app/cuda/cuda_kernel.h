#pragma once
#include <cstdint>
#include <driver_types.h>


void transpose_kernel_invoker(float* src, int num_bboxes, int num_elements, float* dst, cudaStream_t stream);

void transpose_kernel(float* src, int num_bboxes, int num_elements, float* dst, int edge);


void decode_kernel_invoker( float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray,
    int max_objects, cudaStream_t stream);

void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);