#pragma once
#include<app/model/IModel.h>
#include"yolo.h"
namespace app {

    yolo::yolo() = default;
    yolo::~yolo() {
        dispose();
        IModel::dispose();
    };

    yolo::yolo(const std::string& path) {
        modelpath = path;
        init();
    };
    void yolo::dispose() {
        IModel::dispose();
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(decode_ptr_device);
        cudaFree(transpose_device);
    };

    void yolo::forwork(const cv::Mat& img) {
        cv::Mat floatImg;
        img.convertTo(floatImg, CV_32FC3);
        floatImg /= 255;

        cudaMemcpyAsync(buffers[0], floatImg.data, 3 * height * width * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        bool resute = context->executeV2((void**)buffers);
        assert(resute);


        transposeDevice(buffers[1], output_candidates, num_classes + 4, transpose_device);  //transpose [1 84 8400] convert to [1 8400 84]
        cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream);
        decode_result(transpose_device, output_candidates, num_classes, bbox_conf_thresh, decode_ptr_device, max_objects); //���� cuda
        nms_kernel_invoker(decode_ptr_device, nms_thresh, max_objects);//cuda nms          
        cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + max_objects * 7), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<app::Box> boxes;

        int boxes_count = 0;
        int count = std::min((int)*decode_ptr_host, max_objects);

        for (int i = 0; i < count; i++)
        {
            int basic_pos = 1 + i * 7;
            int keep_flag = decode_ptr_host[basic_pos + 6];
            if (keep_flag == 1)
            {
                boxes_count += 1;
                app::Box box;
                box.left = decode_ptr_host[basic_pos + 0];
                box.top = decode_ptr_host[basic_pos + 1];
                box.right = decode_ptr_host[basic_pos + 2];
                box.bottom = decode_ptr_host[basic_pos + 3];
                box.confidence = decode_ptr_host[basic_pos + 4];
                box.class_label = decode_ptr_host[basic_pos + 5];
                boxes.push_back(box);
            }
        }

        for (int i = 0; i < boxes_count; i++)
        {
            cv::Rect roi_area(boxes[i].left, boxes[i].top, boxes[i].right - boxes[i].left, boxes[i].bottom - boxes[i].top);
            cv::rectangle(img, roi_area, cv::Scalar(0, 255, 0), 2);
            std::string  label_string = std::to_string((int)boxes[i].class_label) + " " + std::to_string(boxes[i].confidence);
            cv::putText(img, label_string, cv::Point(boxes[i].left, boxes[i].top - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imwrite("../image_name.jpg", img);
    };


    void yolo::init() {
        IModel::init();

        cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + max_objects * 7));
        decode_ptr_host = new float[1 + max_objects * 7];

        Dims in_dims = engine->getTensorShape("images");
        height = in_dims.d[2];
        width = in_dims.d[3];

        auto out_dims = engine->getTensorShape("output0");
        auto output_size = 1;

        for (int j = 0; j < out_dims.nbDims; j++) {
            output_size *= out_dims.d[j];
        }

        nms_thresh = 0.3;
        cudaMalloc((void**)&buffers[0], 3 * height * width * sizeof(float));
        cudaMalloc((void**)&buffers[1], output_size * sizeof(float));
        cudaMalloc(&transpose_device, output_size * sizeof(float));
    };
};
