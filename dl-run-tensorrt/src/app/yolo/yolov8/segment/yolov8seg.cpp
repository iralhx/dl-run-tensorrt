#pragma once
#include<app/model/IModel.h>
#include"yolov8seg.h"
#include"yolov8seg_kernel.h"

std::vector<app::Point>* find_edge(cv::Mat segment ,float seg_thresh, float scale_to_predict_x, float scale_to_predict_y, float* matrix);

std::vector<app::Point>* find_edge(cv::Mat segment, float seg_thresh, float scale_to_predict_x, float scale_to_predict_y, float* matrix){

    int thread = seg_thresh * 255;
    std::vector<app::Point>* edge = new std::vector<app::Point>;
    // 进行二值化，将非零像素置为255
    cv::Mat binarySegment;
    cv::threshold(segment, binarySegment, thread, 255, cv::THRESH_BINARY);

    // 查找边缘轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binarySegment, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty())
    {

        size_t maxContourIndex = 0;
        double maxContourArea = cv::contourArea(contours[0]);
        for (size_t i = 1; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > maxContourArea) {
                maxContourIndex = i;
                maxContourArea = area;
            }
        }
        const std::vector<cv::Point>& maxContour = contours[maxContourIndex];
        for (size_t i = 0; i < maxContour.size(); ++i) {
            app::Point point;
            float x, y;
            x = maxContour[i].x/ scale_to_predict_x;
            y = maxContour[i].y/ scale_to_predict_y;
            point.x = matrix[0] * x + matrix[1] * y;
            point.y= matrix[3] * x + matrix[4] * y;
            edge->push_back(point);
        }
    }

    return edge;
};
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

    std::vector<Box>* yolov8seg::forword(cv::Mat& img) {

        float find_latency = 0;
        cudaEvent_t find_start_time;
        cudaEvent_t find_end_time;
        cudaEventCreate(&find_start_time);
        cudaEventCreate(&find_end_time);


        CHECK(cudaEventRecord(start_time));
        affine_matrix afmt;
        cv::Size from(img.cols, img.rows);
        get_affine_martrix(afmt, tergetsize, from);
        cudaMemcpyAsync(affine_matrix_d2i_device, afmt.d2i, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(affine_matrix_i2d_device, afmt.i2d, sizeof(afmt.i2d), cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(cuda_device_img, img.data, img.cols * img.rows * img.channels(), cudaMemcpyHostToDevice, stream);
        CHECK(cudaStreamSynchronize(stream));

        preprocess_kernel_img(cuda_device_img, img.cols, img.rows, buffers[0], width, height, affine_matrix_d2i_device, stream);  // cuda前处理 letter_box
        CHECK(cudaStreamSynchronize(stream));

        CHECK(cudaEventRecord(end_time));
        CHECK(cudaEventSynchronize(end_time));
        float latency = 0;
        CHECK(cudaEventElapsedTime(&latency, start_time, end_time));

        printf("前处理: %.5f ms     ", latency);

        CHECK(cudaEventRecord(start_time));
        bool resute = context->executeV2((void**)buffers);
        CHECK(cudaEventRecord(end_time));
        assert(resute);
        CHECK(cudaEventSynchronize(end_time));

        CHECK(cudaEventElapsedTime(&latency, start_time,end_time));

        printf("预测: %.5f ms     ", latency);
        CHECK(cudaEventRecord(start_time));
        transposeDevice(buffers[2], dim_output.d[1], dim_output.d[2], cuda_transpose);

        CHECK(cudaMemset(decode_ptr_device, 0, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT)));
        memset(decode_ptr_host, 0, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT));

        decode_seg_result(cuda_transpose, output_candidates, num_classes, dim_mask.d[1], bbox_conf_thresh,  affine_matrix_d2i_device, decode_ptr_device, max_objects); //后处理 cuda
        nms_kernel_invoker(decode_ptr_device, nms_thresh, max_objects);//cuda nms          
        cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + max_objects * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream);
        CHECK(cudaStreamSynchronize(stream));

        int boxes_count = 0;
        std::vector<app::Box>* boxes =new std::vector<app::Box>;
        int count = std::min((int)*decode_ptr_host, max_objects);

        float scale_to_predict_x = dim_mask.d[3] / (float)width;
        float scale_to_predict_y = dim_mask.d[2] / (float)height;

        for (int i = 0; i < count; i++)
        {
            float* basic_pos = decode_ptr_host+1 + i * NUM_BOX_ELEMENT;
            int keep_flag = basic_pos[6];
            if (keep_flag == 1)
            {
                CHECK(cudaEventRecord(find_start_time));
                app::Box box;
                box.left = basic_pos [0];
                box.top = basic_pos[1];
                box.right = basic_pos[2];
                box.bottom = basic_pos[3];
                box.confidence = basic_pos[4];
                box.class_label = basic_pos[5];

                float* mask_head_predict = buffers[1];
                float left, top, right, bottom;
                float* i2d = afmt.i2d;
                affine_project(i2d, basic_pos[0], basic_pos[1], &left, &top);
                affine_project(i2d, basic_pos[2], basic_pos[3], &right, &bottom);

                float box_width = right - left;
                float box_height = bottom - top;

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




                    cv::Mat mat(mask_out_height, mask_out_width, CV_8U);
                    memcpy(mat.data, mask_out_host, mask_out_width * mask_out_height);
                    box.segment_point = find_edge(mat , seg_thresh, scale_to_predict_x, scale_to_predict_y, afmt.d2i);


                    //cv::imwrite(std::to_string(i)+ ".jpg", mat);

                }
                boxes->push_back(box);

                CHECK(cudaEventRecord(find_end_time));
                CHECK(cudaEventSynchronize(find_end_time));

                CHECK(cudaEventElapsedTime(&find_latency, find_start_time, find_end_time));

                printf("找轮廓: %.5f ms     ", find_latency);

            }
        }



        CHECK(cudaEventRecord(end_time));

        CHECK(cudaEventSynchronize(end_time));
        CHECK(cudaEventElapsedTime(&latency, start_time,end_time));

        printf("后处理: %.5f ms     ", latency);



        cudaEventDestroy(find_end_time);
        cudaEventDestroy(find_start_time);
        return boxes;

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
        cudaMalloc((void**)&buffers[0], in_dims.d[1] * height * width * sizeof(float));
        cudaMalloc((void**)&buffers[2], output_size * sizeof(float));
        cudaMalloc((void**)&buffers[1], output_mask_size * sizeof(float));
        cudaMalloc(&cuda_transpose, output_size * sizeof(float));
        cudaMalloc(&cuda_device_img, 3 * MAX_IMAGE_INPUT_SIZE_THRESH);
    };
};
