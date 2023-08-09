#pragma once
#include<app/common/cuda_common.h>
#include<stdint.h>
// left, top, right, bottom, confidence, class, keepflag
#define NUM_BOX_ELEMENT  8

__host__ __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy);

namespace app {
	void transposeDevice(float* src, int dim1, int dim2, float* dst);

	void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects);
	void preprocess_kernel_img(uint8_t* src,int src_width, int src_height,float* dst, int dst_width,
		int dst_height,float* d2i, cudaStream_t stream);
}