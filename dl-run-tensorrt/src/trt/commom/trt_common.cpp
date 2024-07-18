#include"trt_common.h"

namespace trt {
	void set_device(int device_id)
	{
		cudaSetDevice(device_id);
	}


	int get_version()
	{
		return NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD;
	}

}