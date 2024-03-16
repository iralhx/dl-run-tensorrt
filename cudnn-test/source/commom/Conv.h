#pragma once
#include "Module.h"


namespace common {

    class Conv2D : public IModule {
    public:
        int batch_size; // ���δ�С
        int in_channle, out_channle,kernel_size;//����ͨ�������ͨ�����˴�С
        int stride, padding, dilation; //���������,����
        int in_width, in_height, out_width, out_height;//�����ȣ��߶ȣ������ȣ��߶�
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

            //��������ĸ߶ȺͿ��
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

        // ǰ�򴫲�
        void forward(const IModuleData& input, IModuleData& output) const override;

        // ���򴫲�
        void backward(const IModuleData& nestLayerOutput, IModuleData& lastLayerOutput) override;
    };
}