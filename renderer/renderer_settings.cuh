#pragma once

#include "helper_math.cuh"
#include "renderer_color.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_AMBIENT_OCCLUSION_SAMPLES 512
#define AMBIENT_OCCLUSION_RANDOM_ROTATIONS 1

//define this to disable debug logs
//#define KERNEL_NO_DEBUG

//=========================================
// RENDERER SETTINGS
//=========================================

namespace kernel
{
//to use doubles when stepping (sampleDepth)
//the color is still accumulated in floats
#ifndef KERNEL_USE_DOUBLE
#define KERNEL_USE_DOUBLE 0
#endif

#if KERNEL_USE_DOUBLE==1
	typedef double real;
	typedef double3 real3;
	typedef double4 real4;
	__host__ __device__ __forceinline__ real3 make_real3(const float3& v)
	{
		return make_double3(v.x, v.y, v.z);
	}
	__host__ __device__ __forceinline__ real3 make_real3(const float4& v)
	{
		return make_double3(v.x, v.y, v.z);
	}
	__host__ __device__ __forceinline__ real3 make_real3(const double4& v)
	{
		return make_double3(v.x, v.y, v.z);
	}
	__host__ __device__ __forceinline__ real3 make_real3(double x, double y, double z)
	{
		return make_double3(x, y, z);
	}
	__host__ __device__ __forceinline__ real4 make_real4(double x, double y, double z, double w)
	{
		return make_double4(x, y, z, w);
	}
	__host__ __device__ __forceinline__ real4 make_real4(const real3& xyz, real w)
	{
		return make_double4(xyz.x, xyz.y, xyz.z, w);
	}
#else
	typedef float real;
	typedef float3 real3;
	typedef float4 real4;
	__host__ __device__ __forceinline__ real3 make_real3(const float3& v)
	{
		return v;
	}
	__host__ __device__ __forceinline__ real3 make_real3(const real4& v)
	{
		return make_float3(v.x, v.y, v.z);
	}
	__host__ __device__ __forceinline__ real3 make_real3(float x, float y, float z)
	{
		return make_float3(x, y, z);
	}
	__host__ __device__ __forceinline__ real4 make_real4(float x, float y, float z, float w)
	{
		return make_float4(x, y, z, w);
	}
	__host__ __device__ __forceinline__ float4 make_real4(const real3& xyz, real w)
	{
		return make_float4(xyz.x, xyz.y, xyz.z, w);
	}
#endif
	
	struct OutputTensor
	{
		enum
		{
			ORDERING = 0 //=ColumnMajor
		};
		float* memory;
		int rows, cols, batches;

		__host__ __device__ int idx(int row, int col, int batch) const
		{
			return row + rows * (col + cols * batch);
		}
		__device__ float& coeff(int row, int col, int batch)
		{
			return memory[idx(row, col, batch)];
		}
	};

	//1-1 copy of renderer::ShadingSettings in settings.h
	struct ShadingSettings
	{
		float3 ambientLightColor;
		float3 diffuseLightColor;
		float3 specularLightColor;
		float specularExponent;
		float3 materialColor;
		float aoStrength;

		///the light direction
		/// renderer: world space
		/// post-shading: screen space
		float3 lightDirection;
	};

	enum VolumeFilterMode
	{
		NEAREST,
		TRILINEAR,
		TRICUBIC,
	};

	/**
	 * \brief Subclasses of DVR rendering.
	 * This defines how the TF is given or what to do with it
	 */
	enum class DvrTfMode
	{
		/**
		 * piecewise linear transfer function.
		 * The standard way to render DVR.
		 */
		PiecewiseLinear,
		/**
		 * Multiple transparent isosurfaces.
		 * The transfer function is a sum of dirac deltas
		 * at the control points of the piecewise TF.
		 */
		MultipleIsosurfaces,
		/**
		 * Both piecewise linear DVR and Multi-Iso at the same time
		 */
		Hybrid,
	};
	
	struct RendererDeviceSettings
	{
		cudaTextureObject_t volumeTexNearest;
		cudaTextureObject_t volumeTexLinear;
		
		float2 screenSize;
		int3 volumeResolution;
		float3 voxelSize;
		int binarySearchSteps;
		float stepsize;
		float normalStepSize;
		float3 eyePos;
		float4 currentViewMatrixInverse[4]; //row-by-row
		float4 currentViewMatrix[4];
		float4 nextViewMatrix[4];
		float4 normalMatrix[4];
		float3 boxMin;
		float3 boxSize;
		int aoSamples;
		float aoRadius;
		float aoBias;
		int4 viewport;
		float isovalue;
		float opacityScaling;
		float minDensity;
		float maxDensity;
		float realMinDensity; //first density where opacity>0. Tighter bound than minDensity which is used to scale the tf
		bool useShading;
		ShadingSettings shading;

		DvrTfMode tfMode;
		cudaTextureObject_t tfTexture;
		TfGpuSettings tfPoints;

		bool enableClipPlane;
		float4 clipPlane; //Ax*Bx*Cz+D=0

		//debug
		bool pixelSelected;
		int2 selectedPixel;
		bool voxelFiltered;
		int3 selectedVoxel;
	};

#ifndef KERNEL_INSTRUMENTATION
#define KERNEL_INSTRUMENTATION 1 //by default use instrumentation so that we have the settings in host code
#endif
	/**
	 * \brief Stores per-thread / per-pixel statistics on the algorithm evaluation
	 */
	struct PerPixelInstrumentation
	{
#if KERNEL_INSTRUMENTATION==1
		int densityFetches; //number of fetches in the density volume
		int tfFetches; //number of fetches in the transfer function texture / control points
		int ddaSteps; //number of DDA steps performed
		int isoIntersections; //number of isosurface intersections performed
		int intervalEval; //number of interval integrations
		int intervalStep; //number of substeps done to do the interval quadrature. intervalStep/intervalEval is the mean sample count per quadrature
		int intervalMaxStep; //maximal number of substeps per quadrature.

		//timings (for now only iso)
		int timeDensityFetch;
		int timeDensityFetch_NumSamples;
		int timePolynomialCreation;
		int timePolynomialCreation_NumSamples;
		int timePolynomialSolution;
		int timePolynomialSolution_NumSamples;
		int timeTotal;
#endif
	};

#if KERNEL_INSTRUMENTATION==1
	/**
	 * Increments the instrumentation parameter 'param'.
	 * Assumes that the instrumentation is called 'instrumentation'
	 */
#define KERNEL_INSTRUMENTATION_INC(param) do {(instrumentation)->param++;} while(false)
#define KERNEL_INSTRUMENTATION_ADD(param,val) do {(instrumentation)->param+=(val);} while(false)
#define KERNEL_INSTRUMENTATION_MAX(param,val) do {(instrumentation)->param=max((instrumentation)->param, (val));} while(false)
#define KERNEL_INSTRUMENTATION_TIME_ADD(param,start) do {(instrumentation)->param+=(clock64()-start); (instrumentation)->param##_NumSamples++;} while(false)
#define KERNEL_INSTRUMENTATION_TIME_ADD0(param,start) do {(instrumentation)->param+=(clock64()-start);} while(false)
#else
#define KERNEL_INSTRUMENTATION_INC(param) (void)0
#define KERNEL_INSTRUMENTATION_ADD(param,val) (void)0
#define KERNEL_INSTRUMENTATION_MAX(param,val) (void)0
#define KERNEL_INSTRUMENTATION_TIME_ADD(param,start) (void)0
#define KERNEL_INSTRUMENTATION_TIME_ADD0(param,start) (void)0
#endif

	enum TextureSelection
	{
		PointSampledTexture,
		LinearSampledTexture
	};

}