#include "findline.h"
#include<common/common.h>
#include<trt/builder/trt_builder.h>
#include <NvInfer.h>

using namespace nvinfer1;

namespace trt {


    findline::findline(const std::string& path) {
		assert(common::fileExists(path));
		modelpath = path;
        findline::init();
	};

    cv::Mat findline::forwork(const cv::Mat& img) {
        assert(context != nullptr);
        assert(!img.empty());
        //Ԥ����ͼƬ
        cv::Mat floatImg;
        img.convertTo(floatImg, CV_32FC1);
        floatImg /= 255;

        cudaMemcpyAsync(buffers[0], floatImg.data, IMAGESIZE, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        bool resute = context->executeV2((void**)buffers);
        assert(resute);
        cudaMemcpyAsync(cudahost, buffers[1], IMAGESIZE, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        // �� cudahost �е����ݱ���Ϊͼ��
        cv::Mat resultImage(256, 256, CV_32FC1, cudahost);
        resultImage *= 255;
        return resultImage;
    };

    void findline::dispose()
	{
        context->destroy();
        engine->destroy();
        runtime->destroy();
		cudaStreamDestroy(stream);
		cudaFreeHost(cudahost);
		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
	};

    void findline::init() {


        //��ȡģ��
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
        runtime =createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine=runtime->deserializeCudaEngine(trtModelStreamDet, size);
        assert(engine != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStreamDet;

        //�����ڴ�
        cudaMalloc((void**)&buffers[0], IMAGESIZE);
        cudaMalloc((void**)&buffers[1], IMAGESIZE);
        cudaStreamCreate(&stream);
        cudaMallocHost((void**)&cudahost, IMAGESIZE);
    };

};
