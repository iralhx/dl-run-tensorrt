#pragma once
#include<app/find_line/findline.h>

extern "C" __declspec(dllexport) IModel<cv::Mat>* _cdecl create_findline(const char* path) 
{
	return new app::findline(path);
}

extern "C" __declspec(dllexport) void _cdecl dispose_findline(IModel<cv::Mat>* model)
{
	delete model;
}

extern "C" __declspec(dllexport) uchar* _cdecl findline_forwork(IModel<cv::Mat>*model, cv::Mat & img)
{
	return model->forwork(img).ptr();
}

