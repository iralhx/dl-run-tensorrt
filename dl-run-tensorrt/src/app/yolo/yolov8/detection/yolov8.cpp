#pragma once
#include<app/model/IModel.h>
#include"yolov8.h"
#include"yolov8_kernel.h"
namespace app {


    yolov8::yolov8() = default;
    yolov8::~yolov8() {
        dispose();
        IModel::dispose();
    };

    yolov8::yolov8(const std::string& path) {
        modelpath = path;
        init();
    };
    void yolov8::dispose() {
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(decode_ptr_device);
        cudaFree(affine_matrix_d2i_device);
        cudaFree(cuda_device_img);
        cudaFree(cuda_transpose);
        cudaFree(decode_ptr_host);
        IModel::dispose();
    };

    std::vector<Box> yolov8::forwork(cv::Mat& img) {
        affine_matrix afmt;
        cv::Size from(img.cols, img.rows);
        get_affine_martrix(afmt, tergetsize, from);
        cudaMemcpyAsync(affine_matrix_d2i_device, afmt.d2i, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(cuda_device_img, img.data, img.cols * img.rows * 3, cudaMemcpyHostToDevice, stream);
        CHECK(cudaStreamSynchronize(stream));

        preprocess_kernel_img(cuda_device_img, img.cols, img.rows, buffers[0], width, height, affine_matrix_d2i_device, stream);  // cuda前处理 letter_box
        CHECK(cudaStreamSynchronize(stream));

        bool resute = context->executeV2((void**)buffers);
        assert(resute);

        transposeDevice(buffers[1], num_classes+4, output_candidates, cuda_transpose);


        decode_result(cuda_transpose, output_candidates, num_classes, bbox_conf_thresh, affine_matrix_d2i_device, decode_ptr_device, max_objects); //后处理 cuda
        nms_kernel_invoker(decode_ptr_device, nms_thresh, max_objects);//cuda nms          
        cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream);
        CHECK(cudaStreamSynchronize(stream));

        std::vector<app::Box> boxes;

        int boxes_count = 0;
        int count = std::min((int)*decode_ptr_host, max_objects);

        for (int i = 0; i < count; i++)
        {
            int basic_pos = 1 + i * NUM_BOX_ELEMENT;
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
        return boxes;
    };


    void yolov8::init() {
        IModel::init();

        cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6);
        cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT));
        decode_ptr_host = new float[1 + max_objects * NUM_BOX_ELEMENT];

        Dims in_dims = engine->getTensorShape("images");
        height = in_dims.d[2];
        width = in_dims.d[3];

        cv::Size  tergetsize (width,height );
        this->tergetsize = tergetsize;
        auto out_dims = engine->getTensorShape("output0");
        auto output_size = 1;

        for (int j = 0; j < out_dims.nbDims; j++) {
            output_size *= out_dims.d[j];
        }

        nms_thresh = 0.3;
        cudaMalloc((void**)&buffers[0], 3 * height* width * sizeof(float));
        cudaMalloc((void**)&buffers[1], output_size * sizeof(float));
        cudaMalloc(&cuda_transpose, output_size * sizeof(float));
        cudaMalloc(&cuda_device_img, 3 * MAX_IMAGE_INPUT_SIZE_THRESH);
    };
};
