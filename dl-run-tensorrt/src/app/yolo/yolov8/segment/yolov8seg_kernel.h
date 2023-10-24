#pragma once
#include<app/common/app_cuda_common.h>
#include<stdint.h>
#include<app/yolo/decode_kernel.h>

namespace app {
	void decode_seg_result(float* predict, int num_bboxes, int num_classes,int num_mask, float confidence_threshold, float* invert_affine_matrix,
		float* parray, int max_objects);
	void decode_single_mask(float left, float top, float* mask_weights, float* mask_predict,
		int mask_width, int mask_height, unsigned char* mask_out,
		int mask_dim, int out_width, int out_height, cudaStream_t stream);
}