#pragma once
#include "Module.h"

class Conv2D : public IModule {
public:

    // 前向传播
    void forward(const IModuleData& input, IModuleData& output) const override;

    // 反向传播
    void backward(const IModuleData& nestLayerOutput, IModuleData& lastLayerOutput) override;
};