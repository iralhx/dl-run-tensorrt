#pragma once
#include<app/model/IModel.h>
#include"yolo.h"


struct affine_matrix  //前处理仿射变换矩阵和逆矩阵
{
    float i2d[6];   //仿射变换正矩阵
    float d2i[6];   //仿射变换逆矩阵

};

void get_affine_martrix(affine_matrix& afmt, cv::Size& to, cv::Size& from)  //计算放射变换的正矩阵和逆矩阵
{
    float scale = std::min(to.width / (float)from.width, to.height / (float)from.height);
    afmt.i2d[0] = scale;
    afmt.i2d[1] = 0;
    afmt.i2d[2] = (-scale * from.width + to.width) * 0.5;
    afmt.i2d[3] = 0;
    afmt.i2d[4] = scale;
    afmt.i2d[5] = (-scale * from.height + to.height) * 0.5;
    cv::Mat  cv_i2d(2, 3, CV_32F, afmt.i2d);
    cv::Mat  cv_d2i(2, 3, CV_32F, afmt.d2i);
    cv::invertAffineTransform(cv_i2d, cv_d2i);         //通过opencv获取仿射变换逆矩阵
    memcpy(afmt.d2i, cv_d2i.ptr<float>(0), sizeof(afmt.d2i));
}
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
    };

    void yolo::forwork(cv::Mat& img) {
        affine_matrix afmt;
        cv::Size from(img.cols, img.rows);
        get_affine_martrix(afmt, tergetsize, from);
        int a= img.type();
        memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));
        cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream);

        cudaMemsetAsync(cuda_host_img, 0, sizeof(int), stream);
        memcpy(cuda_host_img, img.data, img.cols * img.rows * 3);
        cudaMemcpyAsync(cuda_device_img, cuda_host_img, img.cols * img.rows * 3, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        float* buffer_idx = (float*)buffers[0];
        preprocess_kernel_img(cuda_device_img, img.cols, img.rows, buffer_idx, width, height, affine_matrix_d2i_device, stream);  // cuda前处理 letter_box
        cudaStreamSynchronize(stream);

        bool resute = context->executeV2((void**)buffers);
        assert(resute);


        cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream);
        decode_result(buffers[1], output_candidates, num_classes, bbox_conf_thresh, affine_matrix_d2i_device, decode_ptr_device, max_objects); //后处理 cuda
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

        cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6);
        cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6);
        cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + max_objects * 7));
        decode_ptr_host = new float[1 + max_objects * 7];

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
        cudaMalloc(&cuda_device_img, 3 * MAX_IMAGE_INPUT_SIZE_THRESH * sizeof(float));
        cudaMallocHost(&cuda_host_img, 3 * MAX_IMAGE_INPUT_SIZE_THRESH * sizeof(float));
    };
};
