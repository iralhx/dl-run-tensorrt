#include "Segformer.h"
#include "Segformer_kernel.h"
#include <algorithm>

namespace app {

	
	
	Segformer::Segformer(const std::string& path) {
		modelpath = path;
		init();
	}
	Segformer::~Segformer() {

		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		cudaFree(cuda_img);
		free(result_img);

	};
	void Segformer::init() {
		IModel::init();

        Dims in_dims = engine->getTensorShape("input");
        height = in_dims.d[2];
        width = in_dims.d[3];
        dim_output = engine->getTensorShape("output");
		outsize = 1;
		out_height = in_dims.d[2];
		out_width = in_dims.d[3];
        for (int j = 0; j < dim_output.nbDims; j++) {
			outsize *= dim_output.d[j];
        }

        cudaMalloc(&buffers[0], in_dims.d[1] * height * width * sizeof(float));
		cudaMalloc(&buffers[1], outsize * sizeof(float));
		result_img = new float[outsize];
		cudaMalloc(&cuda_img, in_dims.d[1] * height * width * sizeof(float));
	}

	class Logger :public ILogger {
		void log(Severity severity, const char* msg) noexcept override {
			if (severity < Severity::kWARNING)
				std::cout << msg << std::endl;
		};
	};
	cv::Mat Segformer::forword(cv::Mat& img)
	{
		CHECK(cudaEventRecord(start_time));
		cv::Mat input;

		cv::resize(img, input, cv::Size(height, width));

		cudaMemcpyAsync(cuda_img, input.data, 3 * height * width, cudaMemcpyHostToDevice, stream);
		CHECK(cudaStreamSynchronize(stream));


		process_image(cuda_img, (float*)buffers[0], width, height,stream);
		CHECK(cudaStreamSynchronize(stream));

		float* proImg = new float[3 * height * width];
		cudaMemcpyAsync(proImg, buffers[0], 3 * height * width, cudaMemcpyDeviceToHost, stream);
		CHECK(cudaStreamSynchronize(stream));


		bool resute = context->executeV2((void**)buffers);
		assert(resute);

		CHECK(cudaMemcpyAsync(result_img, buffers[1], outsize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CHECK(cudaStreamSynchronize(stream));


		CHECK(cudaEventRecord(end_time));

		CHECK(cudaEventSynchronize(end_time));
		float latency = 0;
		CHECK(cudaEventElapsedTime(&latency, start_time, end_time));

		printf("×ÜÊ±¼ä: %.5f ms     \n", latency);
		cv::Mat* mat = new cv::Mat(height, width, CV_32FC1);
		memcpy(mat->data, result_img, outsize * sizeof(float));
		return *mat;
	}
}

