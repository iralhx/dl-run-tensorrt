#pragma once
#include <iostream>
#include<common/common.h>

namespace trt {

	template<typename _T>
	static void destroy_nvidia_pointer(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	bool onnx2trt(const std::string& file, const std::string& savefile);

	void* readmodel(const std::string& modelfile);
}





