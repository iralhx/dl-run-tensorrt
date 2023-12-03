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
		cudaMalloc(&buffers[1], outsize * sizeof(uint32_t));
		result_img = new uint32_t[outsize];
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
		cv::Mat input;

		cv::resize(img, input, cv::Size(height, width));

		cudaMemcpyAsync(cuda_img, input.data, 3 * height * width, cudaMemcpyHostToDevice, stream);
		CHECK(cudaStreamSynchronize(stream));


		process_imgage(cuda_img, (float*)buffers[0], width, height,stream);
		CHECK(cudaStreamSynchronize(stream));

		bool resute = context->executeV2((void**)buffers);
		assert(resute);

		CHECK(cudaMemcpyAsync(result_img, buffers[1], outsize * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
		CHECK(cudaStreamSynchronize(stream));
		uint8_t* im = new uint8_t[outsize];
		int max_value = 0 ,max_index =-1;
		printf("\n");
		for (int i = 0; i < outsize; ++i) 
		{
			im[i] = static_cast<uint8_t>(result_img[i]);
			if (result_img[i]> max_value)
			{
				max_value = result_img[i];
				max_index = i;
			}
		}

		printf("max value %d  ,max index %d \n", max_value, max_index);

		cv::Mat* mat = new cv::Mat(height, width, CV_8UC1);
		memcpy(mat->data, im, outsize);
		cv::imwrite("result1.jpg", *mat);
		return *mat;
	}
}

