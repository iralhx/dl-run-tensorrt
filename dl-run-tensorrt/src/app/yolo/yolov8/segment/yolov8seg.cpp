#pragma once
#include<app/model/IModel.h>
#include"yolov8seg.h"
#include"yolov8seg_kernel.h"

namespace app {

    yolov8seg::yolov8seg() = default;
    yolov8seg::~yolov8seg() {
        dispose();
    };

    yolov8seg::yolov8seg(const std::string& path) {
        modelpath = path;
        init();
    };
    void yolov8seg::dispose() {
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaFree(decode_ptr_device);
        cudaFree(affine_matrix_d2i_device);
        cudaFree(cuda_device_img);
        cudaFree(cuda_transpose);
        cudaFree(decode_ptr_host);
    };

    void yolov8seg::forwork(cv::Mat& img) {
        affine_matrix afmt;
        cv::Size from(img.cols, img.rows);
        get_affine_martrix(afmt, tergetsize, from);
        cudaMemcpyAsync(affine_matrix_d2i_device, afmt.d2i, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(affine_matrix_i2d_device, afmt.i2d, sizeof(afmt.i2d), cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(cuda_device_img, img.data, img.cols * img.rows * 3, cudaMemcpyHostToDevice, stream);
        CHECK(cudaStreamSynchronize(stream));

        preprocess_kernel_img(cuda_device_img, img.cols, img.rows, buffers[0], width, height, affine_matrix_d2i_device, stream);  // cuda前处理 letter_box
        CHECK(cudaStreamSynchronize(stream));

        bool resute = context->executeV2((void**)buffers);
        assert(resute);

        transposeDevice(buffers[2], dim_output.d[1], dim_output.d[2], cuda_transpose);


        decode_seg_result(cuda_transpose, output_candidates, num_classes, dim_mask.d[1], bbox_conf_thresh,  affine_matrix_d2i_device, decode_ptr_device, max_objects); //后处理 cuda
        nms_kernel_invoker(decode_ptr_device, nms_thresh, max_objects);//cuda nms          
        cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream);
        CHECK(cudaStreamSynchronize(stream));

        int boxes_count = 0;
        std::vector<app::Box> boxes;
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
        cv::imwrite("../result.jpg", img);



        for (int i = 0; i < count; i++)
        {
            float* basic_pos = decode_ptr_host+1 + i * NUM_BOX_ELEMENT;
            int keep_flag = basic_pos[6];
            if (keep_flag == 1)
            {
                float* mask_head_predict = buffers[1];
                float left, top, right, bottom;
                float* i2d = afmt.i2d;
                affine_project(i2d, basic_pos[0], basic_pos[1], &left, &top);
                affine_project(i2d, basic_pos[2], basic_pos[3], &right, &bottom);

                float box_width = right - left;
                float box_height = bottom - top;

                float scale_to_predict_x = dim_mask.d[3] / (float)width;
                float scale_to_predict_y = dim_mask.d[2] / (float)height;
                int mask_out_width = box_width * scale_to_predict_x + 0.5f;
                int mask_out_height = box_height * scale_to_predict_y + 0.5f;
                int row_index = basic_pos[7];
                float* mask_weights = cuda_transpose + row_index * dim_output.d[1] +
                    num_classes + 4;

                if (mask_out_width > 0 && mask_out_height > 0) {

                    unsigned char* mask_out_device;
                    unsigned char* mask_out_host;
                    cudaMalloc(&mask_out_device, mask_out_width * mask_out_height);
                    cudaMallocHost(&mask_out_host, mask_out_width * mask_out_height);


                    checkKernel( decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y,
                        mask_weights,
                        mask_head_predict,
                        dim_mask.d[3], dim_mask.d[2], mask_out_device,
                        dim_mask.d[1], mask_out_width, mask_out_height, stream));
                    CHECK(cudaStreamSynchronize(stream));
                    checkRuntime(cudaMemcpyAsync(mask_out_host, mask_out_device,
                        mask_out_width * mask_out_height,
                        cudaMemcpyDeviceToHost, stream));
                    CHECK(cudaStreamSynchronize(stream));

                    cv::Mat seg(mask_out_height, mask_out_width, CV_8U, mask_out_host);
                    cv::imwrite("../"+std::to_string(i) + ".jpg", seg);

                }




            }
        }
       

    };


    void yolov8seg::init() {
        IModel::init();

        cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6);
        cudaMalloc(&affine_matrix_i2d_device, sizeof(float) * 6);
        cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT));
        decode_ptr_host = new float[1 + max_objects * NUM_BOX_ELEMENT];

        Dims in_dims = engine->getTensorShape("images");
        height = in_dims.d[2];
        width = in_dims.d[3];

        cv::Size  tergetsize(width, height);
        this->tergetsize = tergetsize;
        dim_output = engine->getTensorShape("output0");
        auto output_size = 1;

        for (int j = 0; j < dim_output.nbDims; j++) {
            output_size *= dim_output.d[j];
        }

        dim_mask = engine->getTensorShape("output1");

        auto output_mask_size = 1;
        for (int j = 0; j < dim_mask.nbDims; j++) {
            output_mask_size *= dim_mask.d[j];
        }

        nms_thresh = 0.3;
        cudaMalloc((void**)&buffers[0], 3 * height * width * sizeof(float));
        cudaMalloc((void**)&buffers[2], output_size * sizeof(float));
        cudaMalloc((void**)&buffers[1], output_mask_size * sizeof(float));
        cudaMalloc(&cuda_transpose, output_size * sizeof(float));
        cudaMalloc(&cuda_device_img, 3 * MAX_IMAGE_INPUT_SIZE_THRESH);
    };
};
