#pragma once

#include <iostream>
#include <NvInferRuntimeCommon.h>

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



static void set_device(int device_id)
{
	cudaSetDevice(device_id);
}