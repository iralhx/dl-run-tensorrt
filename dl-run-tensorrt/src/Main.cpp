#include<fstream>
#include<iostream>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include <cassert>
#include<trt/commom/trt_common.h>
#include<app/find_line/findline.h>
#include<app/yolov8/yolo.h>


int main()
{

	trt::set_device(0);
	std::string onnx_file = "./../yolov8n.onnx";
	std::string save_file = "./../yolov8n.engine";
	std::string imagefile = "./../bus1.jpg";
	if (!common:: fileExists(save_file))
	{
		trt:: onnx2trt(onnx_file, save_file);
	}


	//worker(save_file, imagefile);

	app::yolo yolo(save_file);



	cv::Mat img =  cv::imread(imagefile);

	 yolo.forwork(img);
	//cv::imshow("Image with Rectangle", mat);
	//cv::waitKey(0);
	//cv::imwrite("./../result1.jpg", mat);
	
	return 0;
}