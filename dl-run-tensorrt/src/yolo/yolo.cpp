#include<yolo/yolo.h>
#include "cuda_runtime_api.h"
#include<NvInferRuntime.h>


#define IMAGESIZE 256*256* sizeof(float)
using namespace nvinfer1;
using namespace common;
using namespace trt;


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

    std::shared_ptr<IRuntime> runtime_det(createInferRuntime( gLogger), destroy_nvidia_pointer<IRuntime>);
    assert(runtime_det != nullptr);
    std::shared_ptr<ICudaEngine> engine_det(runtime_det->deserializeCudaEngine(trtModelStreamDet, size), destroy_nvidia_pointer<ICudaEngine>);
    assert(engine_det != nullptr);
    std::shared_ptr<IExecutionContext> context_det(engine_det->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
    assert(context_det != nullptr);
    

	cv::Mat img = cv::imread(imagefile,cv::ImreadModes::IMREAD_GRAYSCALE);
    assert(!img.empty());
    cv::Mat floatImg;
    img.convertTo(floatImg, CV_32FC1);
    floatImg /= 255;

    // Create GPU buffers on device
    float* buffers[2];
    cudaMalloc((void**)&buffers[0], IMAGESIZE );
    cudaMalloc((void**)&buffers[1], IMAGESIZE );
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint8_t* img_host = nullptr;
    // prepare input data cache in pinned memory 
    cudaMallocHost((void**)&img_host, IMAGESIZE );

    cudaMemcpyAsync(buffers[0], floatImg.data, IMAGESIZE , cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    bool resute =(*context_det).executeV2((void**)buffers);
    assert(resute);
    cudaMemcpyAsync(img_host, buffers[1], IMAGESIZE, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // 将 img_host 中的数据保存为图像
    cv::Mat resultImage(256,256, CV_32FC1, img_host);
    resultImage *= 255;
    cv::imwrite("./../result.jpg", resultImage);

    // 释放内存
    cudaFreeHost(img_host);
}

