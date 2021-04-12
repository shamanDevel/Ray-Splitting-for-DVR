#include "visualizer_kernels.h"
#include "helper_math.cuh"
#include <cuMat/src/Context.h>

#include <errors.h>
#include <inpainting.h>
#include <warping.h>

__device__ unsigned int rgbaToInt(float r, float g, float b, float a)
{
	r = clamp(r*255, 0.0f, 255.0f);
	g = clamp(g*255, 0.0f, 255.0f);
	b = clamp(b*255, 0.0f, 255.0f);
	a = clamp(a*255, 0.0f, 255.0f);
	return (unsigned int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
	//return 0xff000000 | (int(b) << 16) | (int(g) << 8) | int(r);
}
__device__ unsigned int float4ToInt(float4 rgba)
{
	return rgbaToInt(rgba.x, rgba.y, rgba.z, rgba.w);
}
__device__ float4 intToFloat4(unsigned int rgba)
{
	return make_float4(
		(rgba & 0xff) / 255.0f,
		((rgba >> 8) & 0xff) / 255.0f,
		((rgba >> 16) & 0xff) / 255.0f,
		((rgba >> 24) & 0xff) / 255.0f
	);
}

__device__ inline float fetchChannel(
	renderer::OutputTensor input,
	int channel, int x, int y)
{
	if (channel == -1) return 0;
	if (channel == -2) return 1;
	return input.coeff(y, x, channel, -1);
}
__global__ void SelectOutputChannelKernel(
	dim3 virtual_size,
	renderer::OutputTensor input,
	unsigned int* output,
	int rId, int gId, int bId, int aId,
	float scaleRGB, float offsetRGB, float scaleA, float offsetA)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
	{
		float r = fetchChannel(input, rId, x, y) * scaleRGB + offsetRGB;
		float g = fetchChannel(input, gId, x, y) * scaleRGB + offsetRGB;
		float b = fetchChannel(input, bId, x, y) * scaleRGB + offsetRGB;
		float a = fetchChannel(input, aId, x, y) * scaleA + offsetA;

		output[y * input.cols() + x] = rgbaToInt(r, g, b, a);
	}
	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::selectOutputChannel(
	const renderer::OutputTensor& inputTensor, GLubyte* outputBuffer,
	int r, int g, int b, int a, 
	float scaleRGB, float offsetRGB, float scaleA, float offsetA)
{
	unsigned width = inputTensor.cols();
	unsigned height = inputTensor.rows();
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, SelectOutputChannelKernel);
	cudaStream_t stream = ctx.stream();
	SelectOutputChannelKernel
		<<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, 
		 inputTensor,
		 reinterpret_cast<unsigned int*>(outputBuffer), r, g, b, a,
		 scaleRGB, offsetRGB, scaleA, offsetA);
	CUMAT_CHECK_ERROR();
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void ScreenSpaceShadingKernel(
	dim3 virtual_size,
	renderer::OutputTensor input,
	unsigned int* output,
	renderer::ShadingSettings settings)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
	{
		// read input
		float mask = input.coeff(y, x, 0, -1);
		float3 normal = make_float3(
			input.coeff(y, x, 1, -1),
			input.coeff(y, x, 2, -1),
			input.coeff(y, x, 3, -1)
		);
		normal = safeNormalize(normal);
		float ao = input.coeff(y, x, 5, -1);

		float3 color = make_float3(0);
		// ambient
		color += settings.ambientLightColor * settings.materialColor;
		// diffuse
		color += settings.diffuseLightColor * settings.materialColor * abs(dot(normal, settings.lightDirection));
		// specular
		//TODO
		// ambient occlusion
		color *= lerp(1, ao, settings.aoStrength);

		output[y * input.cols() + x] = rgbaToInt(color.x, color.y, color.z, mask);
	}
	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::screenShading(const renderer::OutputTensor& inputTensor, GLubyte* outputBuffer, const renderer::ShadingSettings& settings)
{
	CHECK_ERROR(inputTensor.batches() == renderer::IsoRendererOutputChannels,
		"Number of batches in the tensor must be ", renderer::IsoRendererOutputChannels, " but got ", inputTensor.batches());
	unsigned width = inputTensor.cols();
	unsigned height = inputTensor.rows();
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, ScreenSpaceShadingKernel);
	cudaStream_t stream = ctx.stream();
	ScreenSpaceShadingKernel
		<<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size,
		 inputTensor,
		 reinterpret_cast<unsigned int*>(outputBuffer), settings);
	CUMAT_CHECK_ERROR();
	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void FillColorMapKernel(
	dim3 virtualSize,
	cudaSurfaceObject_t surface,
	cudaTextureObject_t tfTexture)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtualSize)

		auto density = x / static_cast<float>(virtualSize.x);
		auto rgbo = tex1D<float4>(tfTexture, density);

		surf2Dwrite(rgbaToInt(rgbo.x, rgbo.y, rgbo.z, 1.0f), surface, x * 4, y);

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::fillColorMap(cudaSurfaceObject_t colorMap, cudaTextureObject_t tfTexture, int width, int height)
{
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, FillColorMapKernel);
	cudaStream_t stream = ctx.stream();
	FillColorMapKernel
		<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
		(cfg.virtual_size, colorMap, tfTexture);
	CUMAT_CHECK_ERROR();
}

FlowTensor kernel::inpaintFlow(const ::renderer::OutputTensor& input, int maskChannel, int flowXChannel,
	int flowYChannel)
{
	CHECK_ERROR(std::abs(flowXChannel - flowYChannel) == 1,
		"flowXChannel and flowYChannel must be consecutive, but are ",
		flowXChannel, " and ", flowYChannel);
	renderer::Inpainting::MaskTensor mask = input.slice(maskChannel);
	renderer::Inpainting::DataTensor flow = input.block(
		0, 0, std::min(flowXChannel, flowYChannel), input.rows(), input.cols(), 2);
	auto out = renderer::Inpainting::fastInpaintFractional(mask, flow);
	return out;
	//return flow;
}

::renderer::OutputTensor kernel::warp(const ::renderer::OutputTensor& input, const FlowTensor& flow)
{
	return renderer::Warping::warp(input, flow);
}

std::pair<float, float> kernel::extractMinMaxDepth(const ::renderer::OutputTensor& input, int depthChannel)
{
	auto depthValue = input.slice(depthChannel); //unevaluated
	float maxDepth = static_cast<float>(depthValue.maxCoeff());
	float minDepth = static_cast<float>((depthValue + (depthValue < 1e-5).cast<float>()).minCoeff());
	return { minDepth, maxDepth };
}

::renderer::OutputTensor kernel::lerp(const ::renderer::OutputTensor& a, const ::renderer::OutputTensor& b, float v)
{
	return (1 - v)*a + v * b;
}

