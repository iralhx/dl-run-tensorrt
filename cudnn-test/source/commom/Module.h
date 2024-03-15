#pragma once
#include<cudnn.h>

class IModuleData {
public:
    virtual ~IModuleData() = default;
};

class IModule {
public:
    virtual ~IModule() = default;

    // 前向传播
    virtual void forward(const IModuleData& input, IModuleData& output) const = 0;

    // 反向传播
    virtual void backward(const IModuleData& nestLayerOutput,  IModuleData& lastLayerOutput) = 0;
};



