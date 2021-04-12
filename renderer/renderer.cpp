#include "renderer.h"
#include <random>
#include <iomanip>
#include <cuMat/src/Errors.h>
#include <cuMat/src/Context.h>
#include <assert.h>

#include "camera.h"
#include "tf_texture_1d.h"
#include "errors.h"

#include "helper_math.cuh"
#include "renderer_settings.cuh"
#include "kernel_launcher.h"
#include "tf_preintegration.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace std
{
	std::ostream& operator<<(std::ostream& o, const float3& v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const float4& v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z << "," << v.w;
		return o;
	}
}

BEGIN_RENDERER_NAMESPACE

//=========================================
// HELPER 
//=========================================

std::tuple<std::vector<float4>, std::vector<float4>> computeAmbientOcclusionParameters(int samples, int rotations)
{
	std::default_random_engine rnd;
	static std::uniform_real_distribution<float> distr(0.0f, 1.0f);
	//samples
	std::vector<float4> aoHemisphere(samples);
	for (int i = 0; i < samples; ++i)
	{
		float u1 = distr(rnd);
		float u2 = distr(rnd);
		float r = std::sqrt(u1);
		float theta = 2 * M_PI * u2;
		float x = r * std::cos(theta);
		float y = r * std::sin(theta);
		float scale = distr(rnd);
		scale = 0.1 + 0.9 * scale * scale;
		aoHemisphere[i] = make_float4(x*scale, y*scale, std::sqrt(1 - u1)*scale, 0);
	}
	//random rotation vectors
	std::vector<float4> aoRandomRotations(rotations*rotations);
	for (int i = 0; i < rotations*rotations; ++i)
	{
		float x = distr(rnd) * 2 - 1;
		float y = distr(rnd) * 2 - 1;
		float linv = 1.0f / sqrt(x*x + y * y);
		aoRandomRotations[i] = make_float4(x*linv, y*linv, 0, 0);
	}

	return std::make_tuple(aoHemisphere, aoRandomRotations);
}



//texture<float, 3, cudaReadModeElementType> float_tex;
//texture<unsigned char, 3, cudaReadModeNormalizedFloat> char_tex;
//texture<unsigned short, 3, cudaReadModeNormalizedFloat> short_tex;





namespace kernel
{
	void setAOContants(std::tuple<std::vector<float4>, std::vector<float4>>& params); //defined in renderer_static.cu or renderer_rtc.cpp
	void setDeviceSettings(const ::kernel::RendererDeviceSettings& settings); //defined in renderer_static.cu or renderer_rtc.cpp
}

//=========================================
// RENDERER_LAUNCHER
//=========================================

static bool endsWith(std::string const& s, std::string const& suffix) {
	return s.size() >= suffix.size() && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

void render_convert_settings(const std::string& kernelName, const RendererArgs& in, const Volume* volume, ::kernel::RendererDeviceSettings& s)
{
	const auto args = &in;
	const Volume::MipmapLevel* data = volume->getLevel(args->mipmapLevel);
	CHECK_ERROR(data != nullptr, "mipmap level must exist");
	
	s.screenSize = make_float2(args->cameraResolutionX, args->cameraResolutionY);
	s.volumeResolution = make_int3(data->sizeX(), data->sizeY(), data->sizeZ());
	s.binarySearchSteps = args->binarySearchSteps;
	s.stepsize = args->stepsize / std::max({ data->sizeX(), data->sizeY(), data->sizeZ() });
	assert(s.stepsize > FLT_MIN);
	s.normalStepSize = 0.5f;
	s.boxSize = make_float3(
		volume->worldSizeX(),
		volume->worldSizeY(),
		volume->worldSizeZ());
	s.voxelSize = s.boxSize / make_float3(s.volumeResolution - make_int3(1, 1, 1));
	s.boxMin = make_float3(-s.boxSize.x / 2, -s.boxSize.y / 2, -s.boxSize.z / 2) - (s.voxelSize * 0.5f);
	s.isovalue = args->isovalue;
	s.aoBias = args->aoBias;
	s.aoRadius = args->aoRadius;
	s.aoSamples = args->aoSamples;
	s.eyePos = args->cameraOrigin;
	s.viewport = args->cameraViewport;
	if (s.viewport.z < 0) s.viewport.z = args->cameraResolutionX;
	if (s.viewport.w < 0) s.viewport.w = args->cameraResolutionY;
	Camera::computeMatrices(
		args->cameraOrigin, args->cameraLookAt, args->cameraUp,
		args->cameraFovDegrees, args->cameraResolutionX, args->cameraResolutionY, args->nearClip, args->farClip,
		s.currentViewMatrix, s.currentViewMatrixInverse, s.normalMatrix);
	static float4 lastViewMatrix[4] = {
		make_float4(1,0,0,0), make_float4(0,1,0,0),
		make_float4(0,0,1,0), make_float4(0,0,0,1) };
	memcpy(s.nextViewMatrix, lastViewMatrix, sizeof(float4) * 4);
	memcpy(lastViewMatrix, s.currentViewMatrix, sizeof(float4) * 4);
	s.opacityScaling = args->opacityScaling;
	s.minDensity = args->minDensity;
	s.maxDensity = args->maxDensity;
	s.useShading = args->dvrUseShading;
	memcpy(&s.shading, &args->shading, sizeof(::kernel::ShadingSettings));
	s.enableClipPlane = args->enableClipPlane;
	s.clipPlane = args->clipPlane;
	
	//debug
	s.pixelSelected = args->pixelSelected;
	s.selectedPixel = args->selectedPixel;
	s.voxelFiltered = args->voxelFiltered;
	s.selectedVoxel = args->selectedVoxel;

	if (args->renderMode == RendererArgs::RenderMode::DVR)
	{
		static TfTexture1D rendererTfTexture;
		static TfPreIntegration tfPreIntegration;
		
		s.tfMode = args->dvrTfMode;
		if (args->dvrTfMode == ::kernel::DvrTfMode::PiecewiseLinear || 
			args->dvrTfMode == ::kernel::DvrTfMode::Hybrid) {
			const auto& tf = std::get<RendererArgs::TfLinear>(args->tf);
			rendererTfTexture.updateIfChanged(
				tf.densityAxisOpacity, tf.opacityAxis, tf.opacityExtraColorAxis,
				tf.densityAxisColor, tf.colorAxis);

			//copy and transform points, so that we don't need the min/max density scaling in the kernel
			const int N = rendererTfTexture.getTfGpuSettings()->numPoints;
			s.tfPoints.numPoints = N;
			s.tfPoints.positions[0] = rendererTfTexture.getTfGpuSettings()->positions[0];
			s.tfPoints.valuesDvr[0] = rendererTfTexture.getTfGpuSettings()->valuesDvr[0];
			s.tfPoints.valuesIso[0] = rendererTfTexture.getTfGpuSettings()->valuesIso[0];
			s.tfPoints.positions[N - 1] = rendererTfTexture.getTfGpuSettings()->positions[N - 1];
			s.tfPoints.valuesDvr[N - 1] = rendererTfTexture.getTfGpuSettings()->valuesDvr[N - 1];
			s.tfPoints.valuesIso[N - 1] = rendererTfTexture.getTfGpuSettings()->valuesIso[N - 1];
			for (int i = 1; i < N - 1; ++i)
			{
				s.tfPoints.positions[i] = rendererTfTexture.getTfGpuSettings()->positions[i]
					* (s.maxDensity - s.minDensity) + s.minDensity;
				s.tfPoints.valuesDvr[i] = rendererTfTexture.getTfGpuSettings()->valuesDvr[i];
				s.tfPoints.valuesIso[i] = rendererTfTexture.getTfGpuSettings()->valuesIso[i];
			}
			s.realMinDensity = 0;
			for (int i = 1; i < N - 1; ++i)
			{
				if (s.tfPoints.valuesDvr[i].w == 0 && s.tfPoints.valuesIso[i].w == 0)
					s.realMinDensity = s.tfPoints.positions[i];
				else
					break;
			}

			int preIntegration = 0;
			if (endsWith(kernelName, "preintegrate 1D"))
				preIntegration = 1;
			else if (endsWith(kernelName, "preintegrate 2D"))
				preIntegration = 2;
			if (preIntegration==0)
				s.tfTexture = rendererTfTexture.getTextureObjectXYZ();
			else
			{
				
				float timeFor1D, timeFor2D;
				tfPreIntegration.update(&s.tfPoints,
					s.stepsize, s.opacityScaling, &timeFor1D, &timeFor2D);
				std::cout << "Update preintegrated TF, 1D in " << timeFor1D <<
					"s, 2D in " << timeFor2D << "s" << std::endl;
				if (preIntegration==1)
					s.tfTexture = tfPreIntegration.get1Dtexture();
				else //TWO_D
					s.tfTexture = tfPreIntegration.get2Dtexture();
			}
		} else if (args->dvrTfMode == ::kernel::DvrTfMode::MultipleIsosurfaces)
		{
			const auto& tf = std::get<RendererArgs::TfMultiIso>(args->tf);
			const int N = std::min(TF_MAX_CONTROL_POINTS-2,
				static_cast<int>(tf.colors.size()));
			s.tfPoints.numPoints = N+2;
			s.tfPoints.positions[0] = -1;
			s.tfPoints.valuesIso[0] = make_float4(0, 0, 0, 0);
			for (int i=0; i<N; ++i)
			{
				s.tfPoints.positions[i+1] = tf.densities[i]
					* (s.maxDensity - s.minDensity) + s.minDensity;
				s.tfPoints.valuesIso[i+1] = tf.colors[i];
			}
			s.tfPoints.positions[N+1] = 2;
			s.tfPoints.valuesIso[N+1] = make_float4(0, 0, 0, 0);
			s.realMinDensity = N>0 ? s.tfPoints.positions[1] : 1.0f;
		}
	}
}

void render_gpu(const std::string& kernelName, const Volume* volume, 
	const RendererArgs* args, OutputTensor& output, cudaStream_t stream, 
	::kernel::PerPixelInstrumentation* instrumentation,
	GlobalInstrumentation* globalInstrumentation)
{
	CHECK_ERROR(output.cols() == args->cameraResolutionX,
		"Expected the number of columns in the output tensor (", output.cols(), ") to be equal to camere resolution along X (", args->cameraResolutionX, ")");
	CHECK_ERROR(output.rows() == args->cameraResolutionY,
		"Expected the number of rows in the output tensor (", output.rows(), ") to be equal to camere resolution along Y (", args->cameraResolutionY, ")");
	if (args->renderMode == RendererArgs::RenderMode::ISO) {
		CHECK_ERROR(output.batches() == IsoRendererOutputChannels,
			"Excepted the number of batches in the output tensor (", output.batches(), ") to be equal to IsoRendererOutputChannels (", IsoRendererOutputChannels, ")");
	}
	else if (args->renderMode == RendererArgs::RenderMode::DVR) {
		CHECK_ERROR(output.batches() == DvrRendererOutputChannels,
			"Excepted the number of batches in the output tensor (", output.batches(), ") to be equal to DvrRendererOutputChannels (", DvrRendererOutputChannels, ")");
	}

	const Volume::MipmapLevel* data = volume->getLevel(args->mipmapLevel);
	CHECK_ERROR(data != nullptr, "mipmap level must exist");
	
	//set settings
	::kernel::RendererDeviceSettings s;
	render_convert_settings(kernelName, *args, volume, s);
	cudaTextureObject_t volume_nearest = data->dataTexNearestGpu();
	cudaTextureObject_t volume_linear = data->dataTexLinearGpu();
	s.volumeTexNearest = volume_nearest;
	s.volumeTexLinear = volume_linear;
	if (globalInstrumentation)
		globalInstrumentation->numControlPoints = s.tfPoints.numPoints;
	renderer::kernel::setDeviceSettings(s);
	
	::kernel::OutputTensor outputTensor{ output.data(), output.rows(), output.cols(), output.batches() };
	KernelLauncher::Instance().launchKernel(kernelName,
		s.viewport.z - s.viewport.x, s.viewport.w - s.viewport.y,
		s, args, data, outputTensor, stream, instrumentation, globalInstrumentation);
	CUMAT_CHECK_ERROR();

	//CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

#if RENDERER_RUNTIME_COMPILATION==0
int64_t initializeRenderer()
{
	auto params = computeAmbientOcclusionParameters(MAX_AMBIENT_OCCLUSION_SAMPLES, AMBIENT_OCCLUSION_RANDOM_ROTATIONS);
	kernel::setAOContants(params);
	return 1;
}
#else
int64_t initializeRenderer()
{
	auto params = computeAmbientOcclusionParameters(MAX_AMBIENT_OCCLUSION_SAMPLES, AMBIENT_OCCLUSION_RANDOM_ROTATIONS);
	kernel::setAOContants(params);
	return 1;
}
#endif


END_RENDERER_NAMESPACE
