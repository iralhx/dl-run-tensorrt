#pragma once
#include<app/find_line/findline.h>
#include<app/yolo/yolov8/detection/yolov8.h>

extern "C" __declspec(dllexport) IModel<cv::Mat>* _cdecl create_findline(const char* path) 
{
	return new app::findline(path);
}


extern "C" __declspec(dllexport) IModel<std::vector<app::Box>>*_cdecl create_yolov8(const char* path)
{
	return new app::yolov8(path);
}


extern "C" __declspec(dllexport) void _cdecl dispose(IModel<cv::Mat>* model)
{
	delete model;
}

extern "C" __declspec(dllexport) uchar* _cdecl forwork(IModel<cv::Mat>*model, cv::Mat & img)
{
	return model->forwork(img).ptr();
}





