#pragma once
#include <app/model/IModel.h>

namespace app {


    class Segformer :public IModel<cv::Mat>
    {
    public:
        int width;
        int height;
        Dims dim_output;
    private:

        /// <summary>
        /// 模型的输入和输出
        /// </summary>
        void* buffers[2];
        uint8_t* cuda_img;
        float* result_img;
        int outsize;
        int out_width;
        int out_height;
    public:
        Segformer(const std::string& path);
        ~Segformer();
        cv::Mat forword(cv::Mat& img);


        void init();

    };
};