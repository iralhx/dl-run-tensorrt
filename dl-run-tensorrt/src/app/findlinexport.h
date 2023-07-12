#pragma once
#include <trt/find_line/findline.h>

extern "C" __declspec(dllexport) trt::IModel<cv::Mat>* _cdecl create_findline(const char* path) 
{
	return new trt::findline(path);
}

extern "C" __declspec(dllexport) void _cdecl dispose_findline(trt::IModel<cv::Mat>*model)
{
	delete model;
}

extern "C" __declspec(dllexport) uchar* _cdecl findline_forwork(trt::IModel<cv::Mat>*model, const cv::Mat & img)
{
	return model->forwork(img).ptr();
}

