#include<iostream>
#include<stdio.h>
#include <cuda_runtime.h>

#define checkCudaRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)
