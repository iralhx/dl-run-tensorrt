#include <trt\builder\trt_builder.h>
#include <fstream>
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <stdio.h>
#include <string>
#include <memory>
#include <vector>
#include <NvOnnxParser.h>
#include <cassert>
#include <common/common.h>
using namespace nvinfer1;
using namespace std;
namespace trt {

	void saveToTrtModel(const char* TrtSaveFileName, IHostMemory* trtModelStream)
	{
		std::ofstream out(TrtSaveFileName, std::ios::binary);
		if (!out.is_open())
		{
			common::INFO("打开文件失败!");
		}
		out.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
		out.close();
	}

	bool onnx2trt(const string& file, const string& savefile) {

		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_nvidia_pointer<IBuilder>);

		shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<IBuilderConfig>);
		builder->platformHasFastFp16();

		const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		shared_ptr<INetworkDefinition> network(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);
		vector<Dims> dims_setup(1);

		shared_ptr <nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger), destroy_nvidia_pointer<nvonnxparser::IParser>);

		int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
		parser->parseFromFile(file.c_str(), verbosity);

		int net_num_layers = network->getNbLayers();

		builder->setMaxBatchSize(1);
		config->setMaxWorkspaceSize(1 << 30);
		ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
		assert(engine);

		IHostMemory* trtModelStream;
		trtModelStream = engine->serialize();//将引擎序列化，保存到文件中


		saveToTrtModel(savefile.c_str(), trtModelStream);
		trtModelStream->destroy();

		return true;
	};

	void* readmodel(const std::string& modelfile) {

		char* trtModelStreamDet{ nullptr };
		size_t size{ 0 };
		std::ifstream file(modelfile, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStreamDet = new char[size];
			assert(trtModelStreamDet);
			file.read(trtModelStreamDet, size);
			file.close();
		}
		std::shared_ptr<nvinfer1::IRuntime> runtime_det(createInferRuntime(gLogger), destroy_nvidia_pointer<nvinfer1::IRuntime>);
		assert(runtime_det != nullptr);
		std::shared_ptr<ICudaEngine> engine_det(runtime_det->deserializeCudaEngine(trtModelStreamDet, size), destroy_nvidia_pointer<ICudaEngine>);
		assert(engine_det != nullptr);
		std::shared_ptr < IExecutionContext> context_det ( engine_det->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
		assert(context_det != nullptr);
		delete[] trtModelStreamDet;
		return context_det.get();
	};

};



