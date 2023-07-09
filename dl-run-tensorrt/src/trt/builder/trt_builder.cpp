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

void saveToTrtModel(const char* TrtSaveFileName, IHostMemory* trtModelStream)
{
	std::ofstream out(TrtSaveFileName, std::ios::binary);
	if (!out.is_open())
	{
		INFO("打开文件失败!");
	}
	out.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
	out.close();
}


bool Compile(const string& file, const string& savefile) {

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

	for (int i = 0; i < net_num_layers; ++i) {
		auto layer = network->getLayer(i);
		auto name = layer->getName();
		auto input0 = layer->getInput(0);
		if (input0 == nullptr) continue;
		fprintf(stderr, "name: %s \r\n", name);
	}

	builder->setMaxBatchSize(2);
	config->setMaxWorkspaceSize(1 << 30);
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	assert(engine);

	IHostMemory* trtModelStream;
	trtModelStream = engine->serialize();//将引擎序列化，保存到文件中


	saveToTrtModel(savefile.c_str(), trtModelStream);
	trtModelStream->destroy();

	return true;
};