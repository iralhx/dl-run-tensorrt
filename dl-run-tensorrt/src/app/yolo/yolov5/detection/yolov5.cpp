#pragma once
#include<app/model/IModel.h>
#include"yolov5.h"
#include"yolov5_kernel.h"
namespace app {


    yolov5::yolov5() = default;
    yolov5::~yolov5() {

    };

    yolov5::yolov5(const std::string& path) {
        modelpath = path;
        init();
    };
    void yolov5::dispose() {
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(decode_ptr_device);
        cudaFree(affine_matrix_d2i_device);
        cudaFree(cuda_device_img);
        cudaFree(decode_ptr_host);
        IModel::dispose();
    };

    std::vector<Box>* yolov5::forword(cv::Mat& img) {
        affine_matrix afmt;
        cv::Size from(img.cols, img.rows);
        get_affine_martrix(afmt, tergetsize, from);
        cudaMemcpyAsync(affine_matrix_d2i_device, afmt.d2i, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cuda_device_img, img.data, img.cols * img.rows * in_dim.d[1], cudaMemcpyHostToDevice, stream);
        CHECK(cudaStreamSynchronize(stream));


        preprocess_kernel_img(cuda_device_img, img.cols, img.rows, buffers[0], width, height, affine_matrix_d2i_device, stream);  // cuda前处理 letter_box
        CHECK(cudaStreamSynchronize(stream));

        bool resute = context->executeV2((void**)buffers);
        assert(resute);

        v5_decode_result(buffers[1], output_candidates, num_classes, bbox_conf_thresh, affine_matrix_d2i_device, decode_ptr_device, max_objects); //后处理 cuda
       
        CHECK(cudaStreamSynchronize(stream));
        nms_kernel_invoker(decode_ptr_device, nms_thresh, max_objects,stream);//cuda nms      
        CHECK(cudaStreamSynchronize(stream));
        cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream);
        CHECK(cudaStreamSynchronize(stream));

        std::vector<app::Box>* boxes  =new std::vector<app::Box>;

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
                boxes->push_back(box);
            }
        }
        return boxes;
    };


    void yolov5::init() {
        IModel::init();

        cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6);
        cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT));
        decode_ptr_host = new float[1 + max_objects * NUM_BOX_ELEMENT];

        in_dim = engine->getTensorShape("images");
        height = in_dim.d[2];
        width = in_dim.d[3];

        cv::Size  tergetsize (width,height );
        this->tergetsize = tergetsize;
        out_dim = engine->getTensorShape("output");
        auto output_size = 1;

        for (int j = 0; j < out_dim.nbDims; j++) {
            output_size *= out_dim.d[j];
        }
        num_classes = out_dim.d[2] - 5;
        output_candidates = out_dim.d[1];
        nms_thresh = 0.3;
        cudaMalloc((void**)&buffers[0], in_dim.d[1] * height* width * sizeof(float));
        cudaMalloc((void**)&buffers[1], output_size * sizeof(float));
        cudaMalloc(&cuda_device_img, in_dim.d[1] * MAX_IMAGE_INPUT_SIZE_THRESH);
    };
};
