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

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };

    typedef std::vector<Box> BoxArray;
};