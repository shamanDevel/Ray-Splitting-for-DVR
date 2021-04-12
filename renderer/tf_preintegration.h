#pragma once
#include <texture_types.h>

#include "renderer_color.cuh"

/**
 * \brief Transfer Function Pre-Integration.
 * Based on "High-Quality Pre-Integrated Volume Rendering
 * Using Hardware-Accelerated Pixel Shading" by
 * Engel, Kraus, Ertl, 2001.
 *
 * It uses the same color space as the input points (kernel::TfGpuSettings),
 * i.e. XYZ.
 */
class TfPreIntegration
{
	const int resolution_;
	const int integrationSteps_;
	
	cudaArray_t cudaArray1D_{ nullptr };
	cudaSurfaceObject_t surfaceObject1D_{ 0 };
	cudaTextureObject_t textureObject1D_{ 0 };

	cudaArray_t cudaArray2D_{ nullptr };
	cudaSurfaceObject_t surfaceObject2D_{ 0 };
	cudaTextureObject_t textureObject2D_{ 0 };

public:
	/**
	 * \brief 
	 * \param resolution the resolution of the textures
	 * \param integrationSteps the number of integration steps to compute the 2D table
	 */
	TfPreIntegration(int resolution = 512, int integrationSteps = 256);
	~TfPreIntegration();

	/**
	 * \brief Updates the pre-integration tables.
	 * \param tfPoints the piecewise linear transfer function
	 * \param stepsize the step size of the integration
	 * \param opacityScaling a scaling factor on the opacity
	 * \param timeFor1D_out if != null, filled with the time in seconds to compute the 1d texture
	 * \param timeFor2D_out if != null, filled with the time in seconds to compute the 2d texture
	 */
	void update(
		const ::kernel::TfGpuSettings* tfPoints, float stepsize, float opacityScaling,
		float* timeFor1D_out, float* timeFor2D_out);
	
	/**
	 * 1D preintegration texture.
	 * Eq. 8+10 from the paper
	 */
	cudaTextureObject_t get1Dtexture() const { return textureObject1D_; }
	/**
	 * 2D preintegration texture
	 * (input-density, output-density) for a
	 * fixed step size
	 */
	cudaTextureObject_t get2Dtexture() const { return textureObject2D_; }
};
