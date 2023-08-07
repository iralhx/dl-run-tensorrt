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
#include<export/findlinexport.h>

int main()
{



	trt::set_device(0);

	std::string onnx_file = "./../yolov8n.onnx";
	std::string save_file = "./../yolov8n.engine";

	std::string onnx_file_seg = "./../yolov8l-seg.onnx";
	std::string save_file_seg = "./../yolov8l-seg.engine";
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
	std::vector<app::Box>* result;
	result =yolo.forword(img);
	int count;
	result =yolov8_forword(&yolo, img,count);
	int b =get_vector_box_size(result);


	//std::vector<app::Box> boxs =  yolo.forword(img);
	for (size_t i = 0; i < count; i++)
	{
		app::Box* b = get_vector_box(result, i);
		//cv::Mat* seg=(cv::Mat*)(b.segment);
		//cv::imwrite(std::to_string(i) + ".jpg", *(seg));
	}
	//cv::imshow("Image with Rectangle", mat);
	//cv::waitKey(0);
	//cv::imwrite("./../result1.jpg", mat);
	
	return 0;
}