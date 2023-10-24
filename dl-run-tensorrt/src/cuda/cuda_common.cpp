#include "cuda_common.h"


int cuda::getCudaRuntimeVersion() {

    int cudaVersion;
    cudaError_t error = cudaRuntimeGetVersion(&cudaVersion);

    if (error == cudaSuccess)
    {
        return cudaVersion;
    }
    return 0;
}