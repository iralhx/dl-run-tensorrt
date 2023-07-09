#include<yolo/yolo.h>
#include<file/file_help.h>
#include "cuda_runtime_api.h"
#include<NvInferRuntime.h>


#define IMAGESIZE 256*256
using namespace nvinfer1;

void worker(const std::string& modelfile, const std::string& imagefile) {
	if (!fileExists(imagefile))
	{
		INFO("图片：%s 不存在。", imagefile);
	}

    char* trtModelStreamDet{ nullptr };
    size_t size{ 0 };
    std::ifstream file(modelfile, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size);
        file.close();
    }

    std::shared_ptr<IRuntime> runtime_det(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
    assert(runtime_det != nullptr);
    std::shared_ptr<ICudaEngine> engine_det(runtime_det->deserializeCudaEngine(trtModelStreamDet, size), destroy_nvidia_pointer<ICudaEngine>);
    assert(engine_det != nullptr);
    std::shared_ptr<IExecutionContext> context_det(engine_det->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
    assert(context_det != nullptr);


	cv::Mat img = cv::imread(imagefile);
    assert(!img.empty());

    float* buffers[2];
    cudaMalloc((void**)&buffers[0], IMAGESIZE * sizeof(float));
    cudaMalloc((void**)&buffers[1], IMAGESIZE * sizeof(float));
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    cudaMallocHost((void**)&img_host, IMAGESIZE);
    // prepare input data cache in device memory
    cudaMalloc((void**)&img_device, IMAGESIZE);

    memcpy(img_host, img.data, IMAGESIZE);
    cudaMemcpyAsync(img_device, img_host, IMAGESIZE, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    (*context_det).enqueueV2((void**)buffers, stream, nullptr);
    cudaMemcpyAsync(img_host, img_device, IMAGESIZE, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // 将 img_host 中的数据保存为图像
    cv::Mat resultImage(IMAGESIZE, 1, CV_8UC1, img_host);
    resultImage *= 255;
    cv::imwrite("./result.jpg", resultImage);

    // 释放内存
    cudaFreeHost(img_host);
    cudaFree(img_device);
}

