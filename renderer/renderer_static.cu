
#if RENDERER_RUNTIME_COMPILATION==0
// only compile in static pre-compilation mode

#if defined(NDEBUG) || !defined(_DEBUG)
#define KERNEL_NO_DEBUG 1
#endif
 
#define DEFINE_KERNEL_REGISTRATION
#include "kernel_launcher.h" //defines REGISTER_KERNEL
#include "renderer_kernels_iso.cuh" //registers the kernels
#include "renderer_kernels_dvr.cuh" //registers the kernels

namespace renderer
{
	namespace kernel
	{
		void setAOContants(std::tuple<std::vector<float4>, std::vector<float4>>& params)
		{
			CUMAT_SAFE_CALL(cudaMemcpyToSymbol(cAOHemisphere, std::get<0>(params).data(), sizeof(float4)*MAX_AMBIENT_OCCLUSION_SAMPLES));
			CUMAT_SAFE_CALL(cudaMemcpyToSymbol(cAORandomRotations, std::get<1>(params).data(), sizeof(float4)*AMBIENT_OCCLUSION_RANDOM_ROTATIONS*AMBIENT_OCCLUSION_RANDOM_ROTATIONS));
		}
		void setDeviceSettings(const ::kernel::RendererDeviceSettings& settings)
		{
			CUMAT_SAFE_CALL(cudaMemcpyToSymbol(cDeviceSettings, &settings, sizeof(::kernel::RendererDeviceSettings)));
		}
	}
}

#endif