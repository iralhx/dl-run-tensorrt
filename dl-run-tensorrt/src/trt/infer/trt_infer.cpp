#include "trt_infer.h"

bool check_runtime(cudaError_t e, const char* call, int line, const char* file) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

void set_device(int device_id) {
	if (device_id == -1)
		return;
	checkCudaRuntime(cudaSetDevice(device_id));
}



