#pragma once

#include "commons.h"
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <variant>
#include <cassert>

#include "renderer_settings.cuh"

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251) // dll export of STL types
#endif

//forward declarations, CUDA does not like json.hpp
namespace nlohmann
{
	template<typename T = void, typename SFINAE = void>
	struct adl_serializer;
	template<template<typename U, typename V, typename... Args> class ObjectType =
		std::map,
		template<typename U, typename... Args> class ArrayType = std::vector,
		class StringType = std::string, class BooleanType = bool,
		class NumberIntegerType = std::int64_t,
		class NumberUnsignedType = std::uint64_t,
		class NumberFloatType = double,
		template<typename U> class AllocatorType = std::allocator,
		template<typename T, typename SFINAE = void> class JSONSerializer =
		adl_serializer>
		class basic_json;
	using json = basic_json<>;
}

BEGIN_RENDERER_NAMESPACE

class Camera;

struct MY_API ShadingSettings
{
	float3 ambientLightColor = make_float3(0.1f,0.1f,0.1f);
	float3 diffuseLightColor = make_float3(0.8f,0.8f,0.8f);
	float3 specularLightColor = make_float3(0.1f,0.1f,0.1f);
	float specularExponent = 16;
	float3 materialColor = make_float3(1.0f, 1.0f, 1.0f);
	float aoStrength = 0;

	///the light direction
	/// renderer: world space
	/// post-shading: screen space
	float3 lightDirection = make_float3(0,0,1);
};

struct MY_API RendererArgs
{
	//The mipmap level, 0 means the original level
	int mipmapLevel = 0;
	enum class RenderMode
	{
		ISO, //mask, normal, depth, flow
		DVR //mask(alpha), rgb
	};
	RenderMode renderMode = RenderMode::ISO;
	
	int cameraResolutionX = 512;
	int cameraResolutionY = 512;
	double cameraFovDegrees = 45;
	//Viewport (startX, startY, endX, endY)
	//special values endX=endY=-1 delegate to cameraResolutionX and cameraResolutionY
	int4 cameraViewport = make_int4(0, 0, -1, -1);
	float3 cameraOrigin = make_float3(0, 0, -1);
	float3 cameraLookAt = make_float3(0, 0, 0);
	float3 cameraUp = make_float3(0, 1, 0);

	float nearClip = 0.1f;
	float farClip = 10.0f;

	double isovalue = 0.5;
	int binarySearchSteps = 5;
	double stepsize = 0.5;
	enum class VolumeFilterMode
	{
		TRILINEAR,
		TRICUBIC,
		_COUNT_
	};
	VolumeFilterMode volumeFilterMode = VolumeFilterMode::TRILINEAR;

	int aoSamples = 0;
	double aoRadius = 0.1;
	double aoBias = 1e-4;

	using DvrTfMode = kernel::DvrTfMode;
	DvrTfMode dvrTfMode;
	
	//TF Editor
	struct TfLinear
	{
		std::vector<float> densityAxisOpacity;
		std::vector<float> opacityAxis;
		std::vector<float4> opacityExtraColorAxis;
		std::vector<float> densityAxisColor;
		std::vector<float3> colorAxis;
	};
	struct TfMultiIso
	{
		std::vector<float> densities;
		std::vector<float4> colors;
	};
	std::variant<TfLinear, TfMultiIso> tf;
	float opacityScaling = 1.0f;
	float minDensity = 0.0f;
	float maxDensity = 1.0f;
	//enum class TfPreintegration
	//{
	//	OFF,
	//	ONE_D,
	//	TWO_D
	//};
	//TfPreintegration tfPreintegration = TfPreintegration::OFF;

	//shading
	ShadingSettings shading;
	bool dvrUseShading = false;

	bool enableClipPlane = false;
	float4 clipPlane; //Ax*Bx*Cz+D=0

	//debug
	bool pixelSelected = false;
	int2 selectedPixel = make_int2(0,0);
	bool voxelFiltered = false;
	int3 selectedVoxel = make_int3(0, 0, 0);

	/**
	 * \brief Loads the settings json from the GUI
	 * and places the results in the parameters.
	 *
	 * Note 1: the camera settings are not specified, this has to be done with camera.updateRenderArgs(renderer).
	 * Note 2: the light direction specified in renderer.shading is in local view space.
	 *	       To convert it to global world space, use camera.screenToWorld(shading.lightDirection)
	 * \param settings the RendererArgs settings.
	 * \param basePath the directory containing the settings file, to resolve relative dataset paths
	 * \param renderer the render settings
	 * \param camera the camera
	 * \param filename the filename of the volume to load
	 */
	static void load(const nlohmann::json& settings, const std::filesystem::path& basePath,
		RendererArgs& renderer, Camera& camera, std::string& filename);

	/**
	 * \brief Dumps the settings to json.
	 * Only settings inside of RendererArgs are stored. This means, no camera and volume.
	 */
	nlohmann::json toJson() const;
};

/**
 * Global statistics, as opposed to kernel::PerPixelInstrumentation
 */
struct MY_API GlobalInstrumentation
{
	int numControlPoints = 0;
	//for mesh-based approaches
	int numTriangles = 0;
	int numFragments = 0;
};

END_RENDERER_NAMESPACE

#ifdef _WIN32
#pragma warning( pop )
#endif
