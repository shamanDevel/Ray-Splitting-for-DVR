#include "tf_preintegration.h"

#include <chrono>
#include <cuMat/src/Context.h>

TfPreIntegration::TfPreIntegration(int resolution, int integrationSteps)
	: resolution_(resolution), integrationSteps_(integrationSteps)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); //XYZA

	CUMAT_SAFE_CALL(cudaMallocArray(&cudaArray1D_, &channelDesc, resolution, 0, cudaArraySurfaceLoadStore));
	CUMAT_SAFE_CALL(cudaMallocArray(&cudaArray2D_, &channelDesc, resolution, resolution, cudaArraySurfaceLoadStore));

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaTextureAddressMode(cudaAddressModeClamp);
	texDesc.addressMode[1] = cudaTextureAddressMode(cudaAddressModeClamp);
	texDesc.filterMode = cudaTextureFilterMode(cudaFilterModeLinear);
	texDesc.normalizedCoords = 1;

	//Create the surface and texture object.
	resDesc.res.array.array = cudaArray1D_;
	CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&surfaceObject1D_, &resDesc));
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&textureObject1D_, &resDesc, &texDesc, nullptr));
	resDesc.res.array.array = cudaArray2D_;
	CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&surfaceObject2D_, &resDesc));
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&textureObject2D_, &resDesc, &texDesc, nullptr));
}

TfPreIntegration::~TfPreIntegration()
{
	CUMAT_SAFE_CALL(cudaDestroyTextureObject(textureObject1D_));
	CUMAT_SAFE_CALL(cudaDestroySurfaceObject(surfaceObject1D_));
	CUMAT_SAFE_CALL(cudaFreeArray(cudaArray1D_));

	CUMAT_SAFE_CALL(cudaDestroyTextureObject(textureObject2D_));
	CUMAT_SAFE_CALL(cudaDestroySurfaceObject(surfaceObject2D_));
	CUMAT_SAFE_CALL(cudaFreeArray(cudaArray2D_));
}

static void create1DTable(const ::kernel::TfGpuSettings* tfPoints,
	int resolution, float opacityScaling, std::vector<float4>& table1D)
{
	float4 integral = make_float4(0, 0, 0, 0);
	float lastDensity = 0.0f;
	float4 lastValue = tfPoints->queryDvr(lastDensity, tfPoints->searchInterval(lastDensity));
	for (int i=0; i<resolution; ++i)
	{
		float currentDensity = (float(i) + 0.5f) / float(resolution);
		//rectangle rule
		float4 currentValue = tfPoints->queryDvr(currentDensity, tfPoints->searchInterval(currentDensity));
		float3 currentRGB = make_float3(currentValue);
		float3 lastRGB = make_float3(lastValue);
		integral += (currentDensity - lastDensity) * make_float4(
			0.5f * (lastRGB * lastValue.w* opacityScaling + currentRGB * currentValue.w* opacityScaling),
			0.5f * (lastValue.w + currentValue.w) * opacityScaling
		);
		table1D[i] = integral;
		lastValue = currentValue;
		lastDensity = currentDensity;
	}
}

__constant__::kernel::TfGpuSettings cPreintegrationTfPoints;

__global__ void Compute2DPreintegrationTableKernel(
	dim3 virtualSize, cudaSurfaceObject_t table, 
	float stepsize, float opacityScaling, int resolution, int N)
{
	CUMAT_KERNEL_2D_LOOP(istart, iend, virtualSize)
	{
		float dstart = (static_cast<float>(istart) + 0.5f) / static_cast<float>(resolution);
		float dend = (static_cast<float>(iend) + 0.5f) / static_cast<float>(resolution);
		//Riemann-Sum integration
		float3 rgb_sum = make_float3(0, 0, 0);
		float alpha_sum = 0;
		float h = 1.0f / static_cast<float>(N);
		for (int i=1; i<=N; ++i)
		{
			float omega = i * h;
			float dcurrent = (1 - omega) * dstart + omega * dend;
			float4 value = cPreintegrationTfPoints.queryDvr(dcurrent,
				cPreintegrationTfPoints.searchInterval(dcurrent));
			float3 rgbCurrent = make_float3(value);
			float alphaCurrent = value.w * opacityScaling;

			alpha_sum += alphaCurrent * h * stepsize;
			rgb_sum += h * (rgbCurrent * alphaCurrent * stepsize * expf(-alpha_sum));
		}
		float final_alpha = 1 - expf(-alpha_sum);
		//printf("d=[%.3f, %.3f] -> rgb=(%.3f, %.3f, %.3f), alpha=%.3f\n",
		//	dstart, dend, rgb_sum.x, rgb_sum.y, rgb_sum.z, final_alpha);
		float4 rgba = make_float4(rgb_sum, final_alpha);
		surf2Dwrite(rgba, table, (int)sizeof(float4) * istart, iend, cudaBoundaryModeTrap);
		//surf2Dwrite(rgb_sum.x, table, istart * 16, iend);
		//surf2Dwrite(rgb_sum.y, table, istart * 16 + 4, iend);
		//surf2Dwrite(rgb_sum.z, table, istart * 16 + 8, iend);
		//surf2Dwrite(final_alpha, table, istart * 16 + 12, iend);
	}
	CUMAT_KERNEL_2D_LOOP_END
}

void TfPreIntegration::update(const ::kernel::TfGpuSettings* tfPoints, float stepsize, 
                              float opacityScaling, float* timeFor1D_out, float* timeFor2D_out)
{
	std::chrono::steady_clock::time_point startTime, endTime;
	
	//1D preintegration is done on the CPU
	startTime = std::chrono::steady_clock::now();

	std::vector<float4> table1D(resolution_);
	create1DTable(tfPoints, resolution_, opacityScaling, table1D);
	CUMAT_SAFE_CALL(cudaMemcpyToArray(
		cudaArray1D_, 0, 0, table1D.data(), sizeof(float4) * resolution_, cudaMemcpyHostToDevice));
	
	endTime = std::chrono::steady_clock::now();
	if (timeFor1D_out)
		*timeFor1D_out = std::chrono::duration_cast<
		std::chrono::duration<double>>(endTime - startTime).count();
	
	//2D preintegration is done in CUDA
	
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	startTime = std::chrono::steady_clock::now();
	
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(
		cPreintegrationTfPoints, tfPoints, sizeof(::kernel::TfGpuSettings)));

	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
		resolution_, resolution_, Compute2DPreintegrationTableKernel);
	Compute2DPreintegrationTableKernel
		<<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
		(cfg.virtual_size, surfaceObject2D_, stepsize, opacityScaling, resolution_, integrationSteps_);
	CUMAT_CHECK_ERROR();
	
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	endTime = std::chrono::steady_clock::now();
	if (timeFor2D_out)
		*timeFor2D_out = std::chrono::duration_cast<
		std::chrono::duration<double>>(endTime - startTime).count();
}
