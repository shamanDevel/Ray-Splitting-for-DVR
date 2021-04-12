#include "volume.h"

#include <cuMat/src/Context.h>
#include "renderer_utils.cuh"

BEGIN_RENDERER_NAMESPACE

__global__ void ExtractMinMaxKernel(
	dim3 virtualSize,
	cudaTextureObject_t volumeTexture,
	RENDERER_NAMESPACE ::Volume::Histogram* histogram)
{
	CUMAT_KERNEL_3D_LOOP(x, y, z, virtualSize)

		auto density = tex3D<float>(volumeTexture, x, y, z);

	if (density > 0.0f)
	{
		atomicInc(&histogram->numOfNonzeroVoxels, UINT32_MAX);
	}

	//Since atomicMin and atomicMax does not work with floating point values, trick below can be used.
	//Comparing two non-negative floating point numbers is the same as comparing them as if they are integers.
	atomicMin(reinterpret_cast<int*>(&histogram->minDensity), __float_as_int(density));
	atomicMax(reinterpret_cast<int*>(&histogram->maxDensity), __float_as_int(density));

	CUMAT_KERNEL_3D_LOOP_END
}

__global__ void ExtractHistogramKernel(
	dim3 virtualSize,
	cudaTextureObject_t volumeTexture,
	RENDERER_NAMESPACE ::Volume::Histogram* histogram,
	int numOfBins)
{
	CUMAT_KERNEL_3D_LOOP(x, y, z, virtualSize)

		auto density = tex3D<float>(volumeTexture, x, y, z);
	if (density > 0.0f)
	{
		auto densityWidthResolution = (histogram->maxDensity - histogram->minDensity) / numOfBins;

		auto binIdx = static_cast<int>((density - histogram->minDensity) / densityWidthResolution);

		//Precaution against floating-point errors
		binIdx = binIdx >= numOfBins ? (numOfBins - 1) : binIdx;
		//atomicInc(reinterpret_cast<unsigned int*>(histogram->bins + binIdx, UINT32_MAX));
		atomicAdd(histogram->bins + binIdx, 1.0f / histogram->numOfNonzeroVoxels);
	}

	CUMAT_KERNEL_3D_LOOP_END
}

Volume::Histogram Volume::extractHistogram() const
{
	Volume::Histogram histogram;
	auto data = getLevel(0);

	Volume::Histogram* histogramGpu;
	CUMAT_SAFE_CALL(cudaMalloc(&histogramGpu, sizeof(Volume::Histogram)));
	CUMAT_SAFE_CALL(cudaMemcpy(histogramGpu, &histogram, sizeof(Volume::Histogram), cudaMemcpyHostToDevice));

	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(data->sizeX(), data->sizeY(), data->sizeZ(), ExtractMinMaxKernel);
	cudaStream_t stream = ctx.stream();
	ExtractMinMaxKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, data->dataTexNearestGpu(), histogramGpu);
	CUMAT_CHECK_ERROR();

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	cfg = ctx.createLaunchConfig3D(data->sizeX(), data->sizeY(), data->sizeZ(), ExtractHistogramKernel);
	ExtractHistogramKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, data->dataTexNearestGpu(), histogramGpu, histogram.getNumOfBins());
	CUMAT_CHECK_ERROR();

	CUMAT_SAFE_CALL(cudaMemcpy(&histogram, histogramGpu, sizeof(Volume::Histogram), cudaMemcpyDeviceToHost));
	CUMAT_SAFE_CALL(cudaFree(histogramGpu));

	histogram.maxFractionVal = *std::max_element(std::begin(histogram.bins), std::end(histogram.bins));

	return histogram;
}

template<typename T>
__device__ __forceinline__ T convert(float d);

template<>
__device__ __forceinline__ float convert<float>(float d) { return d; }
template<>
__device__ __forceinline__ uint8_t convert<uint8_t>(float d) { return clamp(int(d*255), 0, 255); }
template<>
__device__ __forceinline__ uint16_t convert<uint16_t>(float d) { return clamp(int(d * 65535), 0, 65535); }

template<typename T, Volume::VolumeFilterMode Mode>
__global__ void ResampleVolumeKernel(
	dim3 virtualSize, dim3 inputSize,
	T* outputTexture)
{
	CUMAT_KERNEL_3D_LOOP(x, y, z, virtualSize)
	{
		float inX = float(x) / virtualSize.x * inputSize.x;
		float inY = float(y) / virtualSize.y * inputSize.y;
		float inZ = float(z) / virtualSize.z * inputSize.z;
		float d = kernel::customTex3D(inX, inY, inZ, kernel::integral_constant<int, Mode>());
		outputTexture[x + virtualSize.x * (y + virtualSize.y * z)] = convert<T>(d);
	}
	CUMAT_KERNEL_3D_LOOP_END
}

std::unique_ptr<Volume> Volume::resample(int3 targetResolution, VolumeFilterMode filterMode) const
{
	if (!getLevel(0)->hasGpuData())
		throw std::runtime_error("data has to be available on the GPU, call copy_to_cpu() first");
	kernel::RendererDeviceSettings s;
	s.volumeTexNearest = getLevel(0)->dataTexNearestGpu();
	s.volumeTexLinear = getLevel(0)->dataTexLinearGpu();
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(cDeviceSettings, &s, sizeof(::kernel::RendererDeviceSettings)));

	cuMat::Context& ctx = cuMat::Context::current();
	cudaStream_t stream = 0;// ctx.stream();
	void* outputDataGpu = ctx.mallocDevice(targetResolution.x * targetResolution.y * targetResolution.z * BytesPerType[type_]);
	dim3 inputSize{ static_cast<unsigned>(getLevel(0)->sizeX()), static_cast<unsigned>(getLevel(0)->sizeY()), static_cast<unsigned>(getLevel(0)->sizeZ()) };
	switch (type_)
	{
	case TypeUChar:
	{
		switch (filterMode)
		{
		case NEAREST:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<uint8_t, NEAREST>);
			ResampleVolumeKernel<uint8_t, NEAREST> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<uint8_t*>(outputDataGpu));
			break;
		}
		case TRILINEAR:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<uint8_t, TRILINEAR>);
			ResampleVolumeKernel<uint8_t, TRILINEAR> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<uint8_t*>(outputDataGpu));
			break;
		}
		case TRICUBIC:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<uint8_t, TRICUBIC>);
			ResampleVolumeKernel<uint8_t, TRICUBIC> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<uint8_t*>(outputDataGpu));
			break;
		}
		}
		break;
	}
	case TypeUShort:
	{
		switch (filterMode)
		{
		case NEAREST:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<uint16_t, NEAREST>);
			ResampleVolumeKernel<uint16_t, NEAREST> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<uint16_t*>(outputDataGpu));
			break;
		}
		case TRILINEAR:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<uint16_t, TRILINEAR>);
			ResampleVolumeKernel<uint16_t, TRILINEAR> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<uint16_t*>(outputDataGpu));
			break;
		}
		case TRICUBIC:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<uint16_t, TRICUBIC>);
			ResampleVolumeKernel<uint16_t, TRICUBIC> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<uint16_t*>(outputDataGpu));
			break;
		}
		}
		break;
	}
	case TypeFloat:
	{
		switch (filterMode)
		{
		case NEAREST:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<float, NEAREST>);
			ResampleVolumeKernel<float, NEAREST> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<float*>(outputDataGpu));
			break;
		}
		case TRILINEAR:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<float, TRILINEAR>);
			ResampleVolumeKernel<float, TRILINEAR> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<float*>(outputDataGpu));
			break;
		}
		case TRICUBIC:
		{
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(targetResolution.x, targetResolution.y, targetResolution.z, ResampleVolumeKernel<float, TRICUBIC>);
			ResampleVolumeKernel<float, TRICUBIC> <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (cfg.virtual_size, inputSize, static_cast<float*>(outputDataGpu));
			break;
		}
		}
		break;
	}
	}
	CUMAT_CHECK_ERROR();

	std::unique_ptr<Volume> output = std::make_unique<Volume>(type_, targetResolution.x, targetResolution.y, targetResolution.z);
	CUMAT_SAFE_CALL(cudaMemcpy(output->getLevel(0)->dataCpu<char>(), outputDataGpu, targetResolution.x * targetResolution.y * targetResolution.z * BytesPerType[type_], cudaMemcpyDeviceToHost));

	ctx.freeDevice(outputDataGpu);
	return output;
}


END_RENDERER_NAMESPACE

