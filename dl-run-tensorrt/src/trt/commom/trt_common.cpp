#include"trt_common.h"

namespace trt {
	void set_device(int device_id)
	{
		cudaSetDevice(device_id);
	};

}