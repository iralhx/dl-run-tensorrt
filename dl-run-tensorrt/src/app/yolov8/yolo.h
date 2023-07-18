#pragma once
#include<common/common.h>
#include <opencv2/opencv.hpp>
#include<string>
#include<trt/builder/trt_builder.h>
#include<trt/commom/trt_common.h>
#include"decode_kernel.h"
#include<app/common/object_detetion.h>
#define MAX_IMAGE_INPUT_SIZE_THRESH 2000 * 2000

namespace app {
    class yolo :public IModel<void>
    {
    private:
        float* affine_matrix_d2i_host ;
        float* affine_matrix_d2i_device;
        float* cuda_host_img;
        float* cuda_device_img;
        float* buffers[2];
        int width;
        int height;
        float* decode_ptr_device;
        cv::Size tergetsize;
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