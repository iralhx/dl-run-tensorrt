#pragma once
#include<common/common.h>
#include <opencv2/opencv.hpp>
#include<string>
#include<trt/builder/trt_builder.h>
#include<trt/commom/trt_common.h>
#include"decode_kernel.h"
#include<app/common/object_detetion.h>


namespace app {
    class yolo :public IModel<void>
    {
    private:
        float* buffers[2];
        int width;
        int height;
        float* transpose_device;
        float* decode_ptr_device;
    public:
        int num_classes = 80;
        int output_candidates = 8400;
        float bbox_conf_thresh = 0.5;
        int max_objects = 1024;
        int nms_thresh = 0.3;
        float* decode_ptr_host;
    public:

        yolo();
        ~yolo();
        yolo(const std::string& path);

        void forwork(const cv::Mat& img);
        void init();

    protected:
        void dispose();
    };
};