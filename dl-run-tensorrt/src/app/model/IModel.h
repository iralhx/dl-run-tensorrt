#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include<NvInfer.h>
#include<NvInferRuntime.h>
#include<common/common.h>
#include<trt/commom/trt_common.h>
#include<fstream>
#include<cuda/cuda_common.h>
using namespace nvinfer1;

template<typename T>
class IModel
{
protected:
	std::string modelpath;
	nvinfer1::IExecutionContext* context;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IRuntime* runtime;
	cudaStream_t stream;
	cudaEvent_t start_time;
	cudaEvent_t end_time;
public:
	IModel() = default;
	IModel(const std::string& path) :
		modelpath(path)
	{
		init();
	};
	~IModel() {
		dispose();
	};
	//自定义的输出
	virtual T forword(cv::Mat& img) = 0;

	virtual void init()
	{
		//读取模型
		char* trtModelStreamDet{ nullptr };
		size_t size{ 0 };
		std::ifstream file(modelpath, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStreamDet = new char[size];
			assert(trtModelStreamDet);
			file.read(trtModelStreamDet, size);
			file.close();
		}
		runtime = createInferRuntime(trt::gLogger);
		assert(runtime != nullptr);
		engine = runtime->deserializeCudaEngine(trtModelStreamDet, size);
		assert(engine != nullptr);
		context = engine->createExecutionContext();
		assert(context != nullptr);
		delete[] trtModelStreamDet;

		cudaEventCreate(&start_time);
		cudaEventCreate(&end_time);
		cudaStreamCreate(&stream);
	};
protected:

	virtual void dispose()
	{
		cudaEventDestroy(start_time);
		cudaEventDestroy(end_time);
		context->destroy();
		engine->destroy();
		runtime->destroy();
		cudaStreamDestroy(stream);
	};
};

