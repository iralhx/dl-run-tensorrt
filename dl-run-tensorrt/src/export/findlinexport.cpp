#include "findlinexport.h"


extern "C" __declspec(dllexport) void _cdecl set_device(int index)
{
	trt::set_device(index);
}


extern "C" __declspec(dllexport) IModel<cv::Mat>*_cdecl create_findline(const char* path)
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

extern "C" __declspec(dllexport) int _cdecl yolov8_segment_weight(app::yolov8seg * model)
{
	return model->max_objects;
}


extern "C" __declspec(dllexport) void _cdecl yolov8_forword(
	IModel<std::vector<app::Box>>* model, cv::Mat img, std::vector<app::Box>&result,int &size)
{
	result = model->forword(img);
	size = result.size();
}




extern "C" __declspec(dllexport) void _cdecl dispose(IModel<cv::Mat>*model)
{
	delete model;
}



extern "C" __declspec(dllexport) int _cdecl get_vector_box_size(std::vector<app::Box>*bos)
{

	return bos->size();
}

extern "C" __declspec(dllexport) app::Box* _cdecl get_vector_box(std::vector<app::Box>*boxs, int index)
{
	app::Box* box = &((*boxs)[index]);

	printf("str:%f", box->left)
		;
	return box;

}



