#include "findline.h"

namespace app {

    findline::findline(const std::string& path) {
        assert(common::fileExists(path));
        modelpath = path;
        findline::init();
    };

    cv::Mat findline::forword(cv::Mat& img) {
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
        IModel::dispose();
		cudaFreeHost(cudahost);
		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
	};

    void findline::init() {
        IModel::init();
        //�����ڴ�
        cudaMalloc((void**)&buffers[0], IMAGESIZE);
        cudaMalloc((void**)&buffers[1], IMAGESIZE);
        cudaMallocHost((void**)&cudahost, IMAGESIZE);
    };

};
