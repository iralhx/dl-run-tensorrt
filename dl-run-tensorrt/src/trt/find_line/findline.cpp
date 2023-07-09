#include "findline.h"
#include<common/common.h>
#include<trt/builder/trt_builder.h>
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

        findline() = default;

        findline(const std::string& path) {
            assert(common::fileExists(path));
            modelpath = path;
        };

        ~findline() {
            dispose();
        };
        void* forwork(const cv::Mat& img) override {

            assert(context == nullptr);
            assert(!img.empty());
            //预处理图片
            cv::Mat floatImg;
            img.convertTo(floatImg, CV_32FC1);
            floatImg /= 255;

            cudaMemcpyAsync(buffers[0], floatImg.data, IMAGESIZE, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            bool resute = context->executeV2((void**)buffers);
            assert(resute);
            cudaMemcpyAsync(cudahost, buffers[1], IMAGESIZE, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            // 将 img_host 中的数据保存为图像
            cv::Mat resultImage(256, 256, CV_32FC1, cudahost);
            resultImage *= 255;
            return &resultImage;
        };

    private:
        void Init() {

            context = static_cast<nvinfer1::IExecutionContext*>(trt::readmodel(modelpath));
            cudaMalloc((void**)&buffers[0], IMAGESIZE);
            cudaMalloc((void**)&buffers[1], IMAGESIZE);
            // Create stream
            cudaStreamCreate(&stream);
            // prepare input data cache in pinned memory 
            cudaMallocHost((void**)&cudahost, IMAGESIZE);

        };


        void dispose()
        {
            context->destroy();
            cudaStreamDestroy(stream);
            cudaFreeHost(cudahost);
            cudaFree(buffers[0]);
            cudaFree(buffers[1]);
        };
    };
};
