#pragma once
#include <opencv2/opencv.hpp>

struct affine_matrix  //ǰ�������任����������
{
    float i2d[6];   //����任������
    float d2i[6];   //����任�����

};

void get_affine_martrix(affine_matrix& afmt, cv::Size& to, cv::Size& from);

