#pragma once
#include<string>
#include<trt/model/IModel.h>
namespace trt {
    class findline :public trt::IModel<void*>
    {
    private:
        std::string modelpath;
        nvinfer1::IExecutionContext* context;
        int IMAGESIZE = 256 * 256;
        float* buffers[2];
        cudaStream_t stream;
        uint8_t* cudahost;
    public:

        findline();

        findline(const std::string& path);

        void* forwork(const cv::Mat& img) override;

    };
};
