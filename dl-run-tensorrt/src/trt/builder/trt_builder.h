#pragma once
#include <iostream>
#include<common/common.h>

namespace trt {

	template<typename _T>
	static void destroy_nvidia_pointer(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	class Logger : public nvinfer1::ILogger {
	public:
		virtual void log(Severity severity, const char* msg) noexcept override {

			if (severity == Severity::kINTERNAL_ERROR) {
				fprintf(stderr, "NVInfer INTERNAL_ERROR: %s", msg);
			}
			else if (severity == Severity::kERROR) {
				fprintf(stderr, "NVInfer: %s", msg);
			}
			else  if (severity == Severity::kWARNING) {
				fprintf(stderr, "NVInfer: %s", msg);
			}
			else  if (severity == Severity::kINFO) {
				fprintf(stderr, "NVInfer: %s", msg);
			}
			else {
				fprintf(stderr, "%s", msg);
			}
		}
	};

	static Logger gLogger;

	bool onnx2trt(const std::string& file, const std::string& savefile);

}





