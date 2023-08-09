#pragma once
#include<app/find_line/findline.h>
#include<app/yolo/yolov8/detection/yolov8.h>
#include<app/yolo/yolov8/segment/yolov8seg.h>
#include<trt/commom/trt_common.h>
#include<app/yolo/yolov5/detection/yolov5.h>


extern "C" __declspec(dllexport) void _cdecl set_device(int index);


extern "C" __declspec(dllexport) bool _cdecl onnx2trt(const char* onnxfile, const char* trtfile);

extern "C" __declspec(dllexport) IModel<cv::Mat>*_cdecl create_findline(const char* path);


extern "C" __declspec(dllexport) IModel<std::vector<app::Box>*>*_cdecl create_yolov5_detetion(const char* path);


extern "C" __declspec(dllexport) IModel<std::vector<app::Box>*>*_cdecl create_yolov8_detetion(const char* path);

extern "C" __declspec(dllexport) IModel<std::vector<app::Box>*>*_cdecl create_yolov8_segment(const char* path);


extern "C" __declspec(dllexport) std::vector<app::Box>* _cdecl yolo_forword(
	IModel<std::vector<app::Box>*>*model, cv::Mat img ,int& size);

extern "C" __declspec(dllexport) void _cdecl dispose(IModel<cv::Mat>*model);

extern "C" __declspec(dllexport) int _cdecl get_vector_box_size(std::vector<app::Box>*bos);

extern "C" __declspec(dllexport) app::Box* _cdecl get_vector_box(std::vector<app::Box>*boxs, int index);

extern "C" __declspec(dllexport) void _cdecl delete_vector_box(std::vector<app::Box>*boxs);


extern "C" __declspec(dllexport) cv::Mat* _cdecl himage_to_mat(unsigned char* r,unsigned char* g,unsigned char* b,
	int height,int weith);

extern "C" __declspec(dllexport) int _cdecl get_vector_point_size(std::vector<app::Point>*points);

extern "C" __declspec(dllexport) void _cdecl copy_vector_point(float* rows, float* cols, std::vector<app::Point>*points);
