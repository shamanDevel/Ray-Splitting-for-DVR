#pragma once

#include "commons.h"

#include <cuda_runtime.h>
#include <vector>
#include <memory>

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251) // dll export of STL types
#endif

namespace kernel { struct TfGpuSettings; }

BEGIN_RENDERER_NAMESPACE

/**
 * \brief The 1D transfer function. 
 * It is assembled from piecewise linear functions into a 1d texture.
 *
 */
class MY_API TfTexture1D
{
public:
	struct GpuData
	{
		int sizeOpacity_{ 0 };
		float* densityAxisOpacity_{ nullptr };
		float* opacityAxis_{ nullptr };

		int sizeColor_{ 0 };
		float* densityAxisColor_{ nullptr };
		//in LAB-space
		float3* colorAxis_{ nullptr };

		int cudaArraySize_{ 0 };
		cudaArray_t cudaArrayXYZ_{ nullptr };
		cudaSurfaceObject_t surfaceObjectXYZ_{ 0 };
		cudaTextureObject_t textureObjectXYZ_{ 0 };

		cudaArray_t cudaArrayRGB_{ nullptr };
		cudaSurfaceObject_t surfaceObjectRGB_{ 0 };
		cudaTextureObject_t textureObjectRGB_{ 0 };
	};

public:
	TfTexture1D(int size = 512);
	TfTexture1D(TfTexture1D&&) = delete;
	TfTexture1D(const TfTexture1D&) = delete;
	~TfTexture1D();

	//This function expects colors in CIELab space and TfTexture1D acts accordingly.
	//Returns true iff changed
	bool updateIfChanged(
		const std::vector<float>& densityAxisOpacity, const std::vector<float>& opacityAxis,
		const std::vector<float4>& opacityExtraColorAxis,
		const std::vector<float>& densityAxisColor, const std::vector<float3>& colorAxis);
	
	cudaTextureObject_t getTextureObjectXYZ() const { return gpuData_.textureObjectXYZ_; }
	cudaTextureObject_t getTextureObjectRGB() const { return gpuData_.textureObjectRGB_; }
	const ::kernel::TfGpuSettings* getTfGpuSettings() const { return tfGpuSettings_; }
	
	static float3 rgbToXyz(const float3& rgb);
	static float3 rgbToLab(const float3& rgb);
	static float3 xyzToRgb(const float3& xyz);
	static float3 labToRgb(const float3& lab);
	
private:
	GpuData gpuData_;
	std::vector<float> densityAxisOpacity_;
	std::vector<float> opacityAxis_;
	std::vector<float4> opacityExtraColorAxis_;
	std::vector<float> densityAxisColor_;
	std::vector<float3> colorAxis_;
	::kernel::TfGpuSettings* tfGpuSettings_;

private:
	/**
	 * Combines and filters the separate control points of color + density
	 * into the unified TfGpuSettings structure.
	 */
	void assembleGpuControlPoints();
	
	void destroy();
};

MY_API void computeCudaTexture(const TfTexture1D::GpuData& gpuData);

END_RENDERER_NAMESPACE

#ifdef _WIN32
#pragma warning( pop )
#endif
