#include<fstream>
#include<iostream>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include<yolo/yolo.h>
#include <cassert>


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

	worker(save_file, imagefile);

	
	return 0;
}