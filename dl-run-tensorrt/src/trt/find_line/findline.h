#pragma once
#include<string>
#include<trt/model/IModel.h>
#include<NvInfer.h>
namespace trt {
    class findline :public trt::IModel<void*>
    {
    private:
        std::string modelpath;
        nvinfer1::IExecutionContext* context;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IRuntime* runtime;
        int IMAGESIZE = 256 * 256 * sizeof(float);
        float* buffers[2];
        cudaStream_t stream;
        uint8_t* cudahost;
    public:

        findline();

        ~findline() {
            dispose();
        }
        findline(const std::string& path);

        void* forwork(const cv::Mat& img);

    private:
        void init();
        void dispose();
    };
};
