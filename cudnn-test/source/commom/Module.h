#pragma once
#include<cudnn.h>

class IModuleData {
public:
    virtual ~IModuleData() = default;
};

class IModule {
public:
    virtual ~IModule() = default;

    // ǰ�򴫲�
    virtual void forward(const IModuleData& input, IModuleData& output) const = 0;

    // ���򴫲�
    virtual void backward(const IModuleData& nestLayerOutput,  IModuleData& lastLayerOutput) = 0;
};



