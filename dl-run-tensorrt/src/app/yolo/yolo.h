#pragma once
#include<common/common.h>
#include <opencv2/opencv.hpp>
#include<string>
#include<trt/builder/trt_builder.h>
#include<trt/commom/trt_common.h>
#include<app/cuda/cuda_kernel.h>

void worker(const std::string& modelfile,const std::string& imagefile);
