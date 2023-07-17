#pragma once
#include<app/common/cuda_common.h>
namespace app {
	void transposeDevice(float* src, int num_bboxes, int num_class, float* dst);
	void decode_result(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects);

	void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects);

}