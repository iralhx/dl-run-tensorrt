#include"yolo.h"


void get_affine_martrix(affine_matrix& afmt, cv::Size& to, cv::Size& from)  //计算放射变换的正矩阵和逆矩阵
{
    float scale = std::min(to.width / (float)from.width, to.height / (float)from.height);
    afmt.i2d[0] = scale;
    afmt.i2d[1] = 0;
    afmt.i2d[2] = (-scale * from.width + to.width) * 0.5;
    afmt.i2d[3] = 0;
    afmt.i2d[4] = scale;
    afmt.i2d[5] = (-scale * from.height + to.height) * 0.5;
    cv::Mat  cv_i2d(2, 3, CV_32F, afmt.i2d);
    cv::Mat  cv_d2i(2, 3, CV_32F, afmt.d2i);
    cv::invertAffineTransform(cv_i2d, cv_d2i);         //通过opencv获取仿射变换逆矩阵
    memcpy(afmt.d2i, cv_d2i.ptr<float>(0), sizeof(afmt.d2i));
}