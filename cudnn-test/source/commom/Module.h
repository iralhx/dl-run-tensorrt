#pragma once
#include "cudnn_common.h"

namespace common {
    class IModuleData {
    public:
        virtual ~IModuleData() = default;
    };

    class IModule {
    public:
        cudnnHandle_t handle;
    public:
        IModule(cudnnHandle_t h) :handle(h) 
        {
        
        }
        virtual ~IModule() = default;

        // ǰ�򴫲�
        virtual void forward(const IModuleData& input, IModuleData& output) const = 0;

        // ���򴫲�
        virtual void backward(const IModuleData& nestLayerOutput, IModuleData& lastLayerOutput) = 0;
    };
}



