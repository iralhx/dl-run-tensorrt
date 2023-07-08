#include<iostream>
#include<file\file_help.h>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>


int main()
{

	std::string onnx_file = "E:/VS/WorkSpa2022/dl-run-tensorrt/bin/x64/Debug/FullConModel.onnx";
	std::string save_file = "E:/VS/WorkSpa2022/dl-run-tensorrt/bin/x64/Debug/FullConModel.engine";
	if (!fileExists(save_file))
	{
		Compile(onnx_file, save_file);
	}





	
	set_device(0);
	std::cout << "Hello World!" << std::endl;
	return 0;
}