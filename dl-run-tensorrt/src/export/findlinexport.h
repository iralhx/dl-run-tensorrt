#pragma once
#include<app/find_line/findline.h>
#include<app/yolo/yolov8/detection/yolov8.h>
#include<app/yolo/yolov8/segment/yolov8seg.h>
#include<trt/commom/trt_common.h>

extern "C" __declspec(dllexport) void _cdecl set_device(int index);


extern "C" __declspec(dllexport) IModel<cv::Mat>*_cdecl create_findline(const char* path);

extern "C" __declspec(dllexport) IModel<std::vector<app::Box>>*_cdecl create_yolov8_detetion(const char* path);

extern "C" __declspec(dllexport) IModel<std::vector<app::Box>>*_cdecl create_yolov8_segment(const char* path);

extern "C" __declspec(dllexport) int _cdecl yolov8_segment_weight(app::yolov8seg * model);

extern "C" __declspec(dllexport) void _cdecl yolov8_forword(
	IModel<std::vector<app::Box>>*model, cv::Mat img, std::vector<app::Box>&result ,int& size);


extern "C" __declspec(dllexport) void _cdecl dispose(IModel<cv::Mat>*model);

extern "C" __declspec(dllexport) int _cdecl get_vector_box_size(std::vector<app::Box>*bos);

extern "C" __declspec(dllexport) app::Box _cdecl get_vector_box(std::vector<app::Box>*boxs, int index);


