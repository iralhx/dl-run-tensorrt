#pragma once
#include<app/common/app_cuda_common.h>
#include<stdint.h>
#include<app/yolo/decode_kernel.h>

namespace app {

	void process_image(uint8_t* input, float* output, int height, int weidth, cudaStream_t stream);
	void post_process_result(uint32_t* input, uint8_t* output, int count, cudaStream_t stream);

}