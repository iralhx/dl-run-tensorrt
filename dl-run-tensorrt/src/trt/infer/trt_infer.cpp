#include "trt_infer.h"
#include<common/common.h>

using namespace common;

bool check_runtime(cudaError_t e, const char* call, int line, const char* file) {
    if (e != cudaSuccess) {
        INFO("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}




