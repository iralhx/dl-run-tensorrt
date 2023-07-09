#include<fstream>
#include<iostream>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include<yolo/yolo.h>
#include <cassert>
#include<trt/find_line/findline.h>


int main()
{

	std::string onnx_file = "E:/VS/WorkSpa2022/dl-run-tensorrt/FullConModel.onnx";
	std::string save_file = "E:/VS/WorkSpa2022/dl-run-tensorrt/FullConModel.engine";
	std::string imagefile = "E:/VS/WorkSpa2022/dl-run-tensorrt/1.bmp";
	if (!common:: fileExists(save_file))
	{
		trt:: onnx2trt(onnx_file, save_file);
	}

	set_device(0);

	trt::findline line(save_file);
	cv::Mat(img) = cv::imread(imagefile);


	void* result = line.forwork(img);
	cv::Mat mat(256, 256, CV_32FC1, result);
	cv::imwrite("E:/VS/WorkSpa2022/dl-run-tensorrt/result1.jpg", mat);
	
	return 0;
}