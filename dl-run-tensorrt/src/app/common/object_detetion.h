#pragma once

#include <vector>

namespace app {

    struct Point {
        int x;
        int y;
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