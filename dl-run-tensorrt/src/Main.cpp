#include<fstream>
#include<iostream>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include<yolo/yolo.h>
#include <cassert>
#include<trt/find_line/findline.h>


int main()
{

	std::string onnx_file = "./../FullConModel.onnx";
	std::string save_file = "./../FullConModel.engine";
	std::string imagefile = "./../1.bmp";
	if (!common:: fileExists(save_file))
	{
		trt:: onnx2trt(onnx_file, save_file);
	}

	set_device(0);

	//worker(save_file, imagefile);

	trt::findline line(save_file);
	cv::Mat(img) = cv::imread(imagefile, cv::ImreadModes::IMREAD_GRAYSCALE);


	cv::Mat mat = line.forwork(img);
	cv::imwrite("./../result1.jpg", mat);
	
	return 0;
}