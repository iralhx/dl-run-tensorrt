#include<fstream>
#include<iostream>
#include<file\file_help.h>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include<yolo/yolo.h>
#include <cassert>


int main()
{

	std::string onnx_file = "E:/VS/WorkSpa2022/dl-run-tensorrt/FullConModel.onnx";
	std::string save_file = "E:/VS/WorkSpa2022/dl-run-tensorrt/FullConModel.engine";
	std::string imagefile = "E:/VS/WorkSpa2022/dl-run-tensorrt/1.bmp";
	if (!fileExists(save_file))
	{
		Compile(onnx_file, save_file);
	}

	set_device(0);

	worker(save_file, imagefile);

	
	std::cout << "Hello World!" << std::endl;
	return 0;
}