#include "testTrapezoidBounds.h"

#include <renderer.h>
#include <renderer_utils.cuh>
#include <renderer_kernels_dvr.cuh>

namespace kernel
{
	__device__ PolyWithBounds* polynomials_device[1];
	__device__ unsigned int* polynomials_counter[1];
	__constant__ unsigned int polynomials_count;
	
	struct DvrIntervalEvaluatorTrapezoid_LogPolynomials : IDvrIntervalEvaluator
	{
		static __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& volPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			const float4 dataEntry = dataPoly(tEntry);
			const float4 dataExit = dataPoly(tExit);

			if (dataEntry.w < 1e-6 && dataExit.w < 1e-6) return; //early out

			//first, extract the polynomials
			const auto absorption = dataPoly.cast(ExtractAbsorption()) * cDeviceSettings.opacityScaling;
			const auto color = dataPoly.cast(ExtractColor());
			//auto emission = absorption * color; //CUDA can't deduce that :(
			const auto emission = mul<3, 3, float3, float>(color, absorption);
			KERNEL_INSTRUMENTATION_INC(intervalEval);

			//absorption can be solved analytically
			auto absorptionInt = absorption.integrate();
			const float absorption0 = absorptionInt(tEntry);
			absorptionInt.coeff[0] -= absorption0;
			absorptionInt = -absorptionInt;
			//now absorptionInt(t1) = -[absorption(t)]_t0^t1
			const float transparency = expf(absorptionInt(tExit));
			oBufferOut = 1 - transparency;

			//build function for emission
			const PolyExpPoly<6, struct float3, 4, float> fullEmission =
				polyExpPoly<6, float3, 4, float>(emission, absorptionInt);

			//log polynomial
			unsigned int i = atomicInc(polynomials_counter[0], 0xffffffffu);
			if (i < polynomials_count)
			{
				PolyWithBounds p{ fullEmission, tEntry, tExit };
				polynomials_device[0][i] = p;
			}
		}
	};
}

static void launchKernel(int screenWidth, int screenHeight, cudaStream_t stream,
	const kernel::RendererDeviceSettings& settings,
	cudaTextureObject_t volume_nearest, cudaTextureObject_t volume_linear,
	kernel::OutputTensor& output, kernel::PerPixelInstrumentation* instrumentation)
{
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(cDeviceSettings, &settings, sizeof(::kernel::RendererDeviceSettings)));
	cuMat::KernelLaunchConfig cfg = cuMat::Context::current().createLaunchConfig2D(
		screenWidth, screenHeight,
		kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorTrapezoid_LogPolynomials, 15>>>);
	kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorTrapezoid_LogPolynomials, 15>>>
	<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, output, instrumentation);
	CUMAT_CHECK_ERROR();
}

std::vector<kernel::PolyWithBounds> launchTrapezoidKernel(
	int screenResolution, const kernel::RendererDeviceSettings& settings,
	cudaTextureObject_t volume_nearest, cudaTextureObject_t volume_linear,
	kernel::OutputTensor& output)
{
	//compute all polynomials
	std::vector<kernel::PolyWithBounds> polynomialsHost;
	kernel::PolyWithBounds* polynomialsDeviceLocal;
	unsigned int* polynomialsCounterLocal;
	unsigned int polynomialsCountLocal = 1 << 12;
	CUMAT_SAFE_CALL(cudaMalloc(&polynomialsCounterLocal, sizeof(unsigned int)));
	bool allPolynomialsCopied = false;
	do
	{
		std::cout << "Try to copy polynomials with a buffer of size " << polynomialsCountLocal << std::endl;
		CUMAT_SAFE_CALL(cudaMalloc(&polynomialsDeviceLocal, sizeof(kernel::PolyWithBounds) * polynomialsCountLocal));
		CUMAT_SAFE_CALL(cudaMemset(polynomialsCounterLocal, 0, sizeof(unsigned int)));
		CUMAT_SAFE_CALL(cudaMemcpyToSymbol(kernel::polynomials_device, &polynomialsDeviceLocal,
			sizeof(kernel::PolyWithBounds*), 0, cudaMemcpyHostToDevice));
		CUMAT_SAFE_CALL(cudaMemcpyToSymbol(kernel::polynomials_counter, &polynomialsCounterLocal,
			sizeof(unsigned int*), 0, cudaMemcpyHostToDevice));
		CUMAT_SAFE_CALL(cudaMemcpyToSymbol(*(&kernel::polynomials_count), &polynomialsCountLocal,
			sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

		launchKernel(screenResolution, screenResolution, 0, settings,
			volume_nearest, volume_linear,
			output, nullptr);

		unsigned int polysWritten = 0;
		CUMAT_SAFE_CALL(cudaMemcpy(&polysWritten, polynomialsCounterLocal,
			sizeof(unsigned int), cudaMemcpyDeviceToHost));
		if (polysWritten < polynomialsCountLocal)
		{
			//yeah, we had enough space
			polynomialsHost.resize(polysWritten);
			CUMAT_SAFE_CALL(cudaMemcpy(polynomialsHost.data(), polynomialsDeviceLocal,
				sizeof(kernel::PolyWithBounds) * polysWritten, cudaMemcpyDeviceToHost));
			allPolynomialsCopied = true;
		}
		else
			polynomialsCountLocal *= 2;

		CUMAT_SAFE_CALL(cudaFree(polynomialsDeviceLocal));
	} while (!allPolynomialsCopied);
	CUMAT_SAFE_CALL(cudaFree(polynomialsCounterLocal));

	return polynomialsHost;
}

