#pragma once
#include<app/common/app_cuda_common.h>
#include<stdint.h>
#include<app/yolo/decode_kernel.h>

namespace app {
	void v5_decode_result(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix,
		float* parray, int max_objects);
}