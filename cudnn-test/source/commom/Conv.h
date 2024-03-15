#pragma once
#include "Module.h"

class Conv2D : public IModule {
public:

    // ǰ�򴫲�
    void forward(const IModuleData& input, IModuleData& output) const override;

    // ���򴫲�
    void backward(const IModuleData& nestLayerOutput, IModuleData& lastLayerOutput) override;
};