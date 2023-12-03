#pragma once
#include<app/common/app_cuda_common.h>
#include<stdint.h>
#include<app/yolo/decode_kernel.h>

namespace app {

	void process_imgage(uint8_t* input, float* output, int height, int weidth, cudaStream_t stream);

}