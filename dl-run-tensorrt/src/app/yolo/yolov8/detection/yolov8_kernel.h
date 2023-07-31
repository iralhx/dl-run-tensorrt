#pragma once
#include<app/common/cuda_common.h>
#include<stdint.h>
#include<app/yolo/yolov8/decode_kernel.h>

namespace app {
	void decode_result(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix,
		float* parray, int max_objects);
}