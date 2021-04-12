#if RENDERER_RUNTIME_COMPILATION==1
// only compile in dynamic compilation mode
#include "kernel_launcher.h"

#include <sstream>

static void throwOnError(CUresult err, const char* file, const int line)
{
	if (err != CUDA_SUCCESS)
	{
		const char* pStr;
		cuGetErrorString(err, &pStr);
		const char* pName;
		cuGetErrorName(err, &pName);
		std::stringstream ss;
		ss << "Cuda error " << pName << " at " << file << ":" << line << " : " << pStr;
		throw std::exception(ss.str().c_str());
	}
}
#define CU_SAFE_CALL( err ) throwOnError( err, __FILE__, __LINE__ )

namespace renderer
{
	namespace kernel
	{
		void setAOContants(std::tuple<std::vector<float4>, std::vector<float4>>& params)
		{
			for (const auto& addr : KernelLauncher::Instance().getConstantAddresses("cAOHemisphere"))
			{
				CU_SAFE_CALL(cuMemcpyHtoD(addr, std::get<0>(params).data(), sizeof(float4)*MAX_AMBIENT_OCCLUSION_SAMPLES));
			}
			for (const auto& addr : KernelLauncher::Instance().getConstantAddresses("cAORandomRotations"))
			{
				CU_SAFE_CALL(cuMemcpyHtoD(addr, std::get<1>(params).data(), sizeof(float4)*AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS));
			}
		}
		void setDeviceSettings(const ::kernel::RendererDeviceSettings& settings)
		{
			for (const auto& addr : KernelLauncher::Instance().getConstantAddresses("cDeviceSettings"))
			{
				CU_SAFE_CALL(cuMemcpyHtoD(addr, &settings, sizeof(::kernel::RendererDeviceSettings)));
			}
		}
	}
}

#endif