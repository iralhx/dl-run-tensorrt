#pragma once
#include<app/find_line/findline.h>
#include<app/yolo/yolov8/detection/yolov8.h>
#include<app/yolo/yolov8/segment/yolov8seg.h>
#include<trt/commom/trt_common.h>

extern "C" __declspec(dllexport) void _cdecl set_device(int index)
{
	trt::set_device(index);
}


extern "C" __declspec(dllexport) IModel<cv::Mat>* _cdecl create_findline(const char* path) 
{
	return new app::findline(path);
}


extern "C" __declspec(dllexport) IModel<std::vector<app::Box>>*_cdecl create_yolov8_detetion(const char* path)
{
	return new app::yolov8(path);
}


extern "C" __declspec(dllexport) IModel<std::vector<app::Box>>*_cdecl create_yolov8_segment(const char* path)
{
	return new app::yolov8seg(path);
}



extern "C" __declspec(dllexport) void _cdecl yolov8_forwork(IModel<cv::Mat>*model, cv::Mat img)
{
	model->forwork(img);
}



extern "C" __declspec(dllexport) void _cdecl dispose(IModel<cv::Mat>*model)
{
	delete model;
}



extern "C" __declspec(dllexport) int _cdecl get_vector_box_size(std::vector<app::Box>* bos)
{
	return bos->size();
}

extern "C" __declspec(dllexport) app::Box* _cdecl get_vector_box(std::vector<app::Box>* boxs, int index)
{
	return (app::Box*)&(boxs[index]);
}



