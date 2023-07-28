#include<fstream>
#include<iostream>
#include<trt\infer\trt_infer.h>
#include<trt\builder\trt_builder.h>
#include <cassert>
#include<trt/commom/trt_common.h>
#include<app/find_line/findline.h>
#include<app/yolov8/yolo.h>
#include<app/yolov8/decode_kernel.h>

int main()
{



	int one = 16;
	int two = 20;
	trt::set_device(0);



	//float cuda_arr[16][20];
	//float cuda_arr1[20][16];
	//float* cuda_host_arr;
	//float* cuda_host_arr1;
	//cudaMallocHost((void**)&cuda_arr, sizeof(float) * one * two);
	//cudaMalloc((void**)&cuda_host_arr, sizeof(float) * one * two);

	//cudaMalloc((void**)&cuda_host_arr1, sizeof(float) * one * two);
	//for (size_t i = 0; i < one; i++)
	//{
	//	for (size_t h = 0; h < two; h++)
	//	{
	//		cuda_arr[i][h] = i * two + h;
	//	}
	//}

	//cudaMemcpy(cuda_host_arr, cuda_arr, sizeof(float) * one * two, cudaMemcpyHostToDevice);

	//app::transposeDevice(cuda_host_arr, one , two, cuda_host_arr1);
	//cudaDeviceSynchronize();

	//cudaMemcpy(cuda_arr1, cuda_host_arr1, sizeof(float) * one * two, cudaMemcpyDeviceToHost );

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