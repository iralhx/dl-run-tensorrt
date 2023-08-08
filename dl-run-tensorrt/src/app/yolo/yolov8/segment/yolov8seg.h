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


    class yolov8seg :public IModel<std::vector<Box>*>
    {
    private:
        float* affine_matrix_d2i_device;
        float* affine_matrix_i2d_device;
        uint8_t* cuda_device_img;
        float* cuda_transpose;
        float* buffers[3];
        int width;
        int height;
        float* decode_ptr_device;
        float* decode_ptr_host;
        cv::Size tergetsize;
        int max_objects = 1024;
        Dims dim_output;
        Dims dim_mask;
        int num_classes = 80;
        int output_candidates = 8400;
    public:
        float bbox_conf_thresh = 0.5;
        float nms_thresh = 0.3;
        float seg_thresh = 0.3;
    public:

        yolov8seg();
        ~yolov8seg();
        yolov8seg(const std::string& path);

        std::vector<Box>* forword(cv::Mat& img);
        void init();

    protected:
        void dispose();
    };
};