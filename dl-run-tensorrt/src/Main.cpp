#include<fstream>
#include<iostream>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include <cassert>
#include<trt/commom/trt_common.h>
#include<app/find_line/findline.h>
#include<app/yolo/yolov8/detection/yolov8.h>
#include<app/yolo/decode_kernel.h>
#include<app/yolo/yolov8/segment/yolov8seg.h>
#include<app/yolo/yolov8/segment/yolov8seg_kernel.h>
#include<export/findlinexport.h>
#include<cuda/cuda_common.h>
#include<app/segformer/Segformer.h>
#include <filesystem>
#include <chrono>


int main()
{
	trt::set_device(0);

	std::string onnx_file = "./../../../model/Segformer.onnx";
	std::string save_file = "./../../../model/Segformer.engine";

	std::string imagefile = "I:/Github/dl-run-tensorrt/a.bmp";
	if (!common:: fileExists(save_file))
	{
		trt:: onnx2trt(onnx_file, save_file);
	}

	cv::Mat img = cv::imread(imagefile,cv::IMREAD_GRAYSCALE);
	cv::Mat input;
	cv::cvtColor(img, input, cv::COLOR_GRAY2BGR);

	app::Segformer seg(save_file);
	for (size_t i = 0; i < 1000; i++)
	{
		cv::Mat result1 = seg.forword(input);
	}

	cv::Mat result =seg.forword(input);
	cv::imshow("Image with Rectangle", result * 100);
	cv::waitKey(0);
	cv::destroyAllWindows();
	
	return 0;
}