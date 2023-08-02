#pragma once

#include <vector>

namespace app {
    struct Box {
        float left;
        float top;
        float right;
        float bottom;
        float confidence;
        int class_label;
        cv::Mat* segment;
        Box() = default;

    };

};