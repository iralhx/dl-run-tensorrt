#pragma once
#include<app/common/cuda_common.h>
namespace app {
	void transposeDevice(float* src, int num_bboxes, int num_class, float* dst);
	void decode_result(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix,
		float* parray, int max_objects);

	void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects);
	void preprocess_kernel_img(float* src,int src_width, int src_height,float* dst, int dst_width,
		int dst_height,float* d2i, cudaStream_t stream);
}