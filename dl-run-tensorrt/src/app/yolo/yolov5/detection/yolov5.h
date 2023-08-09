#pragma once
#include<app/yolo/yolo.h>
#include<common/common.h>
#include <opencv2/opencv.hpp>
#include<string>
#include<trt/builder/trt_builder.h>
#include<trt/commom/trt_common.h>
#include<app/yolo/decode_kernel.h>
#include<app/common/object_detetion.h>
#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000

namespace app {
    


    class yolov5 :public IModel<std::vector<Box>*>
    {
    private:
        float* affine_matrix_d2i_device;
        uint8_t* cuda_device_img;
        float* buffers[2];
        int width;
        int height;
        float* decode_ptr_device;
        float* decode_ptr_host;
        cv::Size tergetsize;
        Dims out_dim;
        Dims in_dim;
        int num_classes = 80;
        int output_candidates = 8400;
    public:
        float bbox_conf_thresh = 0.5;
        int max_objects = 1024;
        float nms_thresh = 0.3;
    public:

        yolov5();
        ~yolov5();
        yolov5(const std::string& path);

        std::vector<Box>* forword(cv::Mat& img);
        void init();

    protected:
        void dispose();
    };
};