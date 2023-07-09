#include<string>
#include<trt/model/IModel.h>
#include<common/common.h>
class findline :public trt::IModel<void>
{
private:
	std::string modelpath;
public:
	findline(const std::string& modelpath) {
		assert(common::fileExists(modelpath));



	};
	~findline();
	void forwork(const cv::Mat& img) override;

};
