#pragma once
#include<app/model/IModel.h>
namespace app {
    class findline :public IModel<cv::Mat>
    {
    private:
        int IMAGESIZE = 256 * 256 * sizeof(float);
        float* buffers[2];
        uint8_t* cudahost;
    public:

        ~findline() {
            dispose();
        }
        findline(const std::string& path);

        cv::Mat forwork(const cv::Mat& img);
        void init() override;

    protected:
        void dispose() override;
    };
};
