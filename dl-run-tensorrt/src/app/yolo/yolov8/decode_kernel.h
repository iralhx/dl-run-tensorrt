#pragma once
#include<app/common/cuda_common.h>
#include<stdint.h>
// left, top, right, bottom, confidence, class, keepflag
#define NUM_BOX_ELEMENT  8

static __host__ __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy) {
	*ox = matrix[0] * x + matrix[1] * y + matrix[2];
	*oy = matrix[3] * x + matrix[4] * y + matrix[5];
}


namespace app {
	void transposeDevice(float* src, int dim1, int dim2, float* dst);

	void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects);
	void preprocess_kernel_img(uint8_t* src,int src_width, int src_height,float* dst, int dst_width,
		int dst_height,float* d2i, cudaStream_t stream);
}