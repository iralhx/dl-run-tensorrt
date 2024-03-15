#include<fstream>
#include<iostream>
#include <cassert>
#include <filesystem>
#include <chrono>
#include<cudnn.h>
#include <string>
#include"commom/Conv.h"

#define CHECK_CUDNN(expression)                             \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

int main()
{
	std::printf("123");
    Conv2D a;

    // 定义神经网络参数
    const int input_channels = 1;
    const int input_height = 28;
    const int input_width = 28;
    const int output_classes = 10;


    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    cudnnTensorDescriptor_t input_desc, output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

    // 设置输入和输出张量描述符
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, input_channels, input_height, input_width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, output_classes, 1, 1));


    // 定义神经网络层描述符
    cudnnActivationDescriptor_t activation_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    /*CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
        data, conv1filterDesc, pconv1, conv1Desc,
        conv1algo, workspace, m_workspaceSize, &beta,
        conv1Tensor, conv1));*/

	return 0;
}