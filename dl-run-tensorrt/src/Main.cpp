#include<fstream>
#include<iostream>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include <cassert>
#include<trt/commom/trt_common.h>
#include<app/find_line/findline.h>
#include<app/yolo/yolov8/detection/yolov8.h>
#include<app/yolo/yolov8/decode_kernel.h>
#include<app/yolo/yolov8/segment/yolov8seg.h>
#include<app/yolo/yolov8/segment/yolov8seg_kernel.h>

int main()
{



	trt::set_device(0);

	std::string onnx_file = "./../yolov8n.onnx";
	std::string save_file = "./../yolov8n.engine";

	std::string onnx_file_seg = "./../yolov8n-seg.onnx";
	std::string save_file_seg = "./../yolov8n-seg.engine";
	std::string imagefile = "./../bus1.jpg";
	if (!common:: fileExists(save_file))
	{
		trt:: onnx2trt(onnx_file, save_file);
	}
	if (!common::fileExists(save_file_seg))
	{
		trt::onnx2trt(onnx_file_seg, save_file_seg);
	}

	//worker(save_file, imagefile);

	app::yolov8seg yolo(save_file_seg);



	cv::Mat img =  cv::imread(imagefile);

	yolo.forwork(img);
	//cv::imshow("Image with Rectangle", mat);
	//cv::waitKey(0);
	//cv::imwrite("./../result1.jpg", mat);
	
	return 0;
}