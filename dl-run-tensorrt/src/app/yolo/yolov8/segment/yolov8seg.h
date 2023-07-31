#pragma once
#include<app/yolo/yolo.h>
#include<common/common.h>
#include <opencv2/opencv.hpp>
#include<string>
#include<trt/builder/trt_builder.h>
#include<trt/commom/trt_common.h>
#include<app/yolo/yolov8/decode_kernel.h>
#include<app/common/object_detetion.h>
#include<app/model/IModel.h>
#define MAX_IMAGE_INPUT_SIZE_THRESH 2000 * 2000

namespace app {


    class yolov8seg :public IModel<void>
    {
    private:
        float* affine_matrix_d2i_device;
        uint8_t* cuda_device_img;
        float* cuda_transpose;
        float* buffers[3];
        int width;
        int height;
        float* decode_ptr_device;
        float* decode_ptr_host;
        cv::Size tergetsize;

        Dims dim_output;
        Dims dim_mask;
    public:
        int num_classes = 80;
        int output_candidates = 8400;
        float bbox_conf_thresh = 0.5;
        int max_objects = 1024;
        int nms_thresh = 0.3;
    public:

        yolov8seg();
        ~yolov8seg();
        yolov8seg(const std::string& path);

        void forwork(cv::Mat& img);
        void init();

    protected:
        void dispose();
    };
};