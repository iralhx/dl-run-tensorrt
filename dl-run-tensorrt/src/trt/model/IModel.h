#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include<NvInfer.h>
#include<NvInferRuntime.h>

namespace trt {

	static void set_device(int device_id)
	{
		cudaSetDevice(device_id);
	}

	template<typename T>
	class IModel
	{
	public:
		//自定义的输出
		virtual T forwork(const cv::Mat& img) = 0;
	};
}
