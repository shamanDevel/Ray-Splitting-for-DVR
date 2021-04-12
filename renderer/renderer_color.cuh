#pragma once

#include "helper_math.cuh"
#include "renderer_commons.cuh"
#include "renderer_math.cuh"

namespace kernel
{
	//From https://github.com/berendeanicolae/ColorSpace/blob/master/src/Conversion.cpp
	//Assumes color channels are in [0,1]
	__device__ __host__
		inline float3 rgbToXyz(const float3& rgb)
	{
		auto r = ((rgb.x > 0.04045f) ? powf((rgb.x + 0.055f) / 1.055f, 2.4f) : (rgb.x / 12.92f)) * 100.0f;
		auto g = ((rgb.y > 0.04045f) ? powf((rgb.y + 0.055f) / 1.055f, 2.4f) : (rgb.y / 12.92f)) * 100.0f;
		auto b = ((rgb.z > 0.04045f) ? powf((rgb.z + 0.055f) / 1.055f, 2.4f) : (rgb.z / 12.92f)) * 100.0f;

		auto x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
		auto y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
		auto z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

		return { x, y, z };
	}

	//Output color channels are in [0,1]
	__device__ __host__
		inline float3 xyzToRgb(const float3& xyz)
	{
		auto x = xyz.x / 100.0f;
		auto y = xyz.y / 100.0f;
		auto z = xyz.z / 100.0f;

		auto r = x * 3.2404542f + y * -1.5371385f + z * -0.4985314f;
		auto g = x * -0.9692660f + y * 1.8760108f + z * 0.0415560f;
		auto b = x * 0.0556434f + y * -0.2040259f + z * 1.0572252f;

		r = (r > 0.0031308f) ? (1.055f * pow(r, 1.0f / 2.4f) - 0.055f) : (12.92f * r);
		g = (g > 0.0031308f) ? (1.055f * pow(g, 1.0f / 2.4f) - 0.055f) : (12.92f * g);
		b = (b > 0.0031308f) ? (1.055f * pow(b, 1.0f / 2.4f) - 0.055f) : (12.92f * b);

		return { r, g, b };
	}

	__device__ __host__
		inline float3 xyzToLab(const float3& xyz)
	{
		auto x = xyz.x / 95.047f;
		auto y = xyz.y / 100.00f;
		auto z = xyz.z / 108.883f;

		x = (x > 0.008856f) ? cbrt(x) : (7.787f * x + 16.0f / 116.0f);
		y = (y > 0.008856f) ? cbrt(y) : (7.787f * y + 16.0f / 116.0f);
		z = (z > 0.008856f) ? cbrt(z) : (7.787f * z + 16.0f / 116.0f);

		return { (116.0f * y) - 16.0f, 500.0f * (x - y), 200.0f * (y - z) };
	}

	__device__ __host__
		inline float3 rgbToLab(const float3& rgb)
	{
		return xyzToLab(rgbToXyz(rgb));
	}
	
	__device__ __host__
		inline float3 labToXyz(const float3& lab)
	{
		auto y = (lab.x + 16.0f) / 116.0f;
		auto x = lab.y / 500.0f + y;
		auto z = y - lab.z / 200.0f;

		auto x3 = x * x * x;
		auto y3 = y * y * y;
		auto z3 = z * z * z;

		x = ((x3 > 0.008856f) ? x3 : ((x - 16.0f / 116.0f) / 7.787f)) * 95.047f;
		y = ((y3 > 0.008856f) ? y3 : ((y - 16.0f / 116.0f) / 7.787f)) * 100.0f;
		z = ((z3 > 0.008856f) ? z3 : ((z - 16.0f / 116.0f) / 7.787f)) * 108.883f;

		return { x, y, z };
	}

	__device__ __host__
		inline float3 labToRgb(const float3& lab)
	{
		return xyzToRgb(labToXyz(lab));
	}


#define TF_MAX_CONTROL_POINTS 32
	/**
	 * This structure stores the control points of the TF
	 * for use in the rendering kernels.
	 */
	struct TfGpuSettings
	{
		//the positions of the control points.
		//The first control point must be ==0, the last one ==1
		float positions[TF_MAX_CONTROL_POINTS];

		//The values at the control points, XYZ opacity
		float4 valuesDvr[TF_MAX_CONTROL_POINTS];

		//The values at the control points, XYZ opacity
		float4 valuesIso[TF_MAX_CONTROL_POINTS];
		
		//the number of control points, must be <= TF_MAX_CONTROL_POINTS
		int numPoints;

		/**
		 * Searches for the interval of control points that contain the
		 * specified density.
		 * \param density the density to search for. Must be in [0,1]
		 * \return the lower index in the arrays of positions and values.
		 * The upper index is simply lower+1
		 */
		__host__ __device__ int searchInterval(float density) const
		{
			int i;
			//for now, a simple linear search is used
			for (i = 0; i < numPoints - 2; ++i)
				if (positions[i + 1] > density) break;
			return i;
		}

		/**
		 * Utility to query the interpolated value at the specified density.
		 *
		 * \param density the density to search for
		 * \param lowerInterval the lower interval index, as computed by searchInterval(float)
		 * \return the interpolated TF value (XYZ + opacity)
		 */
		__host__ __device__ float4 queryDvr(float density, int lowerInterval) const
		{
			const float pLow = positions[lowerInterval];
			const float pHigh = positions[lowerInterval + 1];
//#ifndef KERNEL_NO_DEBUG
//			if (pLow > density || pHigh < density) {
//				printf("  error in color::query(): d=%.4f is not in [%.4f, %.4f] (idx=%d)\n",
//					density, pLow, pHigh, lowerInterval);
//				return values[lowerInterval]; //dummy value
//			}
//#else
			density = clamp(density, pLow, pHigh);
//#endif
			const float frac = (density - pLow) / (pHigh - pLow);
			return (1 - frac) * valuesDvr[lowerInterval] + frac * valuesDvr[lowerInterval + 1];
		}
	};
}
