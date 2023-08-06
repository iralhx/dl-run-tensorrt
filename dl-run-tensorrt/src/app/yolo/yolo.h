#pragma once
#include <opencv2/opencv.hpp>

struct affine_matrix  //前处理仿射变换矩阵和逆矩阵
{
    float i2d[6];   //仿射变换正矩阵
    float d2i[6];   //仿射变换逆矩阵

};

void get_affine_martrix(affine_matrix& afmt, cv::Size& to, cv::Size& from);

