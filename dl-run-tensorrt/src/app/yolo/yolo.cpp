#pragma once
#include<app/model/IModel.h>
#include<app/common/object_detetion.h>
namespace app {
    class yolo :public IModel<BoxArray>
    {
    private:
        int IMAGESIZE = 256 * 256 * sizeof(float);
        float* buffers[2];
        uint8_t* cudahost;
    public:

        ~yolo() {
            dispose();
        }
        yolo(const std::string& path);

        BoxArray forwork(const cv::Mat& img);

    protected:
        void init() override;
        void dispose() override;
    };
};
