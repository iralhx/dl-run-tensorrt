#include "findlinexport.h"



extern "C" __declspec(dllexport) int _cdecl getCudaRuntimeVersion() {
	return cuda::getCudaRuntimeVersion();
}


extern "C" __declspec(dllexport) void _cdecl set_device(int index)
{
	trt::set_device(index);
}


extern "C" __declspec(dllexport) bool _cdecl onnx2trt(const char* onnxfile, const char* trtfile) {
	return trt::onnx2trt(onnxfile, trtfile);
}



extern "C" __declspec(dllexport) IModel<cv::Mat>*_cdecl create_findline(const char* path)
{
	return new app::findline(path);
}

extern "C" __declspec(dllexport) IModel<std::vector<app::Box>*>*_cdecl create_yolov5_detetion(const char* path) {

	return new app::yolov5(path);
}

extern "C" __declspec(dllexport) IModel<std::vector<app::Box>*>*_cdecl create_yolov8_detetion(const char* path)
{
	return new app::yolov8(path);
}


extern "C" __declspec(dllexport) IModel<std::vector<app::Box>*>*_cdecl create_yolov8_segment(const char* path)
{
	return new app::yolov8seg(path);
}


extern "C" __declspec(dllexport) std::vector<app::Box>* _cdecl yolo_forword(
	IModel<std::vector<app::Box>*>* model, cv::Mat img,int &size)
{
	std::vector<app::Box>* result = model->forword(img);
	size = result->size();
	return result;
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
	app::Box* box = &(*boxs)[index];
	
	return box;
}

extern "C" __declspec(dllexport) void _cdecl delete_vector_box(std::vector<app::Box>*boxs) {
	delete boxs;
}


extern "C" __declspec(dllexport) cv::Mat * _cdecl himage_to_mat(unsigned char* r, unsigned char* g, unsigned char* b,
	int height, int weith) {
	cv::Mat* mat=new cv::Mat(height, weith, CV_8UC3);
	for (size_t i = 0; i < height*weith; i++)
	{
		int index = i * 3;
		mat->data[index] = b[i];
		mat->data[index + 1] = g[i];
		mat->data[index + 2] = r[i];
	}
	return mat;
}


extern "C" __declspec(dllexport) int _cdecl get_vector_point_size(std::vector<app::Point>*points) {
	return points->size();
}



extern "C" __declspec(dllexport) void _cdecl copy_vector_point(float* rows, float* cols, std::vector<app::Point>*points) {
	
	for (size_t i = 0; i < points->size(); i++)
	{
		rows[i] = points->at(i).y;
		cols[i] = points->at(i).x;
	}
}




app::Segformer* _cdecl create_segformer(const char* path)
{
	return new app::Segformer(path);
}


float* _cdecl segformer_forword(app::Segformer* model,cv::Mat* img)
{
	return model->forword(*img);
}

void _cdecl delete_segformer(app::Segformer* model)
{
	model->~Segformer();
}


