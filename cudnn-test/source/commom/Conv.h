#pragma once
#include "Module.h"


namespace common {

    class Conv2D : public IModule {
    public:
        int batch_size; // 批次大小
        int in_channle, out_channle,kernel_size;//输入通道，输出通道，核大小
        int stride, padding, dilation; //步长，填充,膨胀
        int in_width, in_height, out_width, out_height;//输入宽度，高度，输出宽度，高度
        cudnnConvolutionDescriptor_t convDesc;

        cudnnFilterDescriptor_t filterDesc;
        cudnnTensorDescriptor_t inputDesc, outputDesc;
    public:
        Conv2D(cudnnHandle_t handle ,int batch_size , int in_channle, int out_channle, int kernel_size,
            int in_width, int in_height,int stride = 1,int padding =0 ,int dilation = 1):IModule(handle),
            batch_size(batch_size), in_channle(in_channle),
            out_channle(out_channle), kernel_size(kernel_size), in_width(in_width),in_height(in_height),
            stride(stride),padding(padding),dilation(dilation)
        {

            //计算输出的高度和宽度
            //https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;


            CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
            CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));

            CHECK_CUDNN(cudnnSetConvolution2dDescriptor( convDesc,
                padding, padding,
                stride, stride,
                dilation, dilation,
                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
            ));
            CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                CUDNN_DATA_FLOAT,
                CUDNN_TENSOR_NCHW,
                out_channle,in_channle, kernel_size, kernel_size));
        }

        // 前向传播
        void forward(const IModuleData& input, IModuleData& output) const override;

        // 反向传播
        void backward(const IModuleData& nestLayerOutput, IModuleData& lastLayerOutput) override;
    };
}