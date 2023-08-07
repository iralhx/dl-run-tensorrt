#pragma once

#include <vector>

namespace app {

    struct Point {
        float x;
        float y;
    };

    struct Box {
        float left;
        float top;
        float right;
        float bottom;
        float confidence;
        int class_label;
        std::vector<Point>* segment_point;
        Box() = default;

    };

};