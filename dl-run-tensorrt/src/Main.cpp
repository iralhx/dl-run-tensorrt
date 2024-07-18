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
#include<cudnn.h>

int main()
{

	size_t cudnn_version =  cudnnGetVersion();

	int trt_version = trt::get_version();

	int version = cuda::getCudaRuntimeVersion();
	std::cout <<"cuda version  :"<< version <<"   cudnn  version  :"<<cudnn_version<<"   trt  version  :"<<trt_version << std::endl;
	trt::set_device(0);

	std::string onnx_file = "I:/Github/dl-run-tensorrt/model/best.onnx";
	const char* save_file = "I:/Github/dl-run-tensorrt/model/red_jiu.engine";

	std::string imagefile = "I:/Github/dl-run-tensorrt/b.bmp";
	if (!common:: fileExists(save_file))
	{
		trt:: onnx2trt(onnx_file, save_file);
	}

	cv::Mat img = cv::imread(imagefile,cv::IMREAD_GRAYSCALE);
	cv::Mat input;
	cv::cvtColor(img, input, cv::COLOR_GRAY2BGR);
	app::yolov8seg y(save_file);
	std::vector<app::Box>* result1 = y.forword(input);
	/*IModel<std::vector<app::Box>*>* de = create_yolov8_detetion( save_file);

 	std::vector<app::Box>* result = de->forword(input);*/
	cv::waitKey(0);
	cv::destroyAllWindows();
	
	return 0;
}
