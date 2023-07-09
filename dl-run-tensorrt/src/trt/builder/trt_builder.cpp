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


static const char* pooling_type_name(nvinfer1::PoolingType type) {
	switch (type) {
	case nvinfer1::PoolingType::kMAX: return "MaxPooling";
	case nvinfer1::PoolingType::kAVERAGE: return "AveragePooling";
	case nvinfer1::PoolingType::kMAX_AVERAGE_BLEND: return "MaxAverageBlendPooling";
	}
	return "Unknow pooling type";
}

static const char* activation_type_name(nvinfer1::ActivationType activation_type) {
	switch (activation_type) {
	case nvinfer1::ActivationType::kRELU: return "ReLU";
	case nvinfer1::ActivationType::kSIGMOID: return "Sigmoid";
	case nvinfer1::ActivationType::kTANH: return "TanH";
	case nvinfer1::ActivationType::kLEAKY_RELU: return "LeakyRelu";
	case nvinfer1::ActivationType::kELU: return "Elu";
	case nvinfer1::ActivationType::kSELU: return "Selu";
	case nvinfer1::ActivationType::kSOFTSIGN: return "Softsign";
	case nvinfer1::ActivationType::kSOFTPLUS: return "Parametric softplus";
	case nvinfer1::ActivationType::kCLIP: return "Clip";
	case nvinfer1::ActivationType::kHARD_SIGMOID: return "Hard sigmoid";
	case nvinfer1::ActivationType::kSCALED_TANH: return "Scaled tanh";
	case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return "Thresholded ReLU";
	}
	return "Unknow activation type";
}

static string layer_type_name(nvinfer1::ILayer* layer) {
	switch (layer->getType()) {
	case nvinfer1::LayerType::kCONVOLUTION: return "Convolution";
	case nvinfer1::LayerType::kFULLY_CONNECTED: return "Fully connected";
	case nvinfer1::LayerType::kACTIVATION: {
		nvinfer1::IActivationLayer* act = (nvinfer1::IActivationLayer*)layer;
		auto type = act->getActivationType();
		return activation_type_name(type);
	}
	case nvinfer1::LayerType::kPOOLING: {
		nvinfer1::IPoolingLayer* pool = (nvinfer1::IPoolingLayer*)layer;
		return pooling_type_name(pool->getPoolingType());
	}
	case nvinfer1::LayerType::kLRN: return "LRN";
	case nvinfer1::LayerType::kSCALE: return "Scale";
	case nvinfer1::LayerType::kSOFTMAX: return "SoftMax";
	case nvinfer1::LayerType::kDECONVOLUTION: return "Deconvolution";
	case nvinfer1::LayerType::kCONCATENATION: return "Concatenation";
	case nvinfer1::LayerType::kELEMENTWISE: return "Elementwise";
	case nvinfer1::LayerType::kPLUGIN: return "Plugin";
	case nvinfer1::LayerType::kUNARY: return "UnaryOp operation";
	case nvinfer1::LayerType::kPADDING: return "Padding";
	case nvinfer1::LayerType::kSHUFFLE: return "Shuffle";
	case nvinfer1::LayerType::kREDUCE: return "Reduce";
	case nvinfer1::LayerType::kTOPK: return "TopK";
	case nvinfer1::LayerType::kGATHER: return "Gather";
	case nvinfer1::LayerType::kMATRIX_MULTIPLY: return "Matrix multiply";
	case nvinfer1::LayerType::kRAGGED_SOFTMAX: return "Ragged softmax";
	case nvinfer1::LayerType::kCONSTANT: return "Constant";
	case nvinfer1::LayerType::kRNN_V2: return "RNNv2";
	case nvinfer1::LayerType::kIDENTITY: return "Identity";
	case nvinfer1::LayerType::kPLUGIN_V2: return "PluginV2";
	case nvinfer1::LayerType::kSLICE: return "Slice";
	case nvinfer1::LayerType::kSHAPE: return "Shape";
	case nvinfer1::LayerType::kPARAMETRIC_RELU: return "Parametric ReLU";
	case nvinfer1::LayerType::kRESIZE: return "Resize";
	}
	return "Unknow layer type";
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
		auto type_str = layer_type_name(layer);
		auto input0 = layer->getInput(0);
		if (input0 == nullptr) continue;

		auto output0 = layer->getOutput(0);
		auto input_dims = input0->getDimensions();
		auto output_dims = output0->getDimensions();

		INFO("name:%s ,type:%s ,index :%d \r\n",name,type_str,i);
	}

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
