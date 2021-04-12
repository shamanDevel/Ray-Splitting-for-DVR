#pragma once

#include "renderer_settings.cuh"
#include "renderer_utils.cuh"
#include "helper_math.cuh"
#include "renderer_math.cuh"

#define IMPLIES(a,b) (!(a) || (b))

#ifndef VOXEL_EVAL_INTERVAL_EXTENDED_EARLY_OUT
#define VOXEL_EVAL_INTERVAL_EXTENDED_EARLY_OUT false
#endif

namespace kernel
{
	//=========================================
	// KERNEL IMPLEMENTATIONS - DVR
	//=========================================
	//
	// DvrKernel (renderer_kernels_dvr.cuh)
	//  |-- DvrFixedStepSize (nearest, trilinear, tricubic)
	//  |
	//  |-- DvrDDA
	//       |- DvrVoxelEvalFixedStepsTexture
	//       |- DvrVoxelEvalFixedStepsControlPoints
	//       |
	//       |- DvrVoxelEvalInterval
	//           |- DvrIntervalEvaluatorSimple
	//           |- DvrIntervalEvaluatorStepping
	//           |- DvrIntervalEvaluatorTrapezoid


	static constexpr float OpacityEarlyOut = 0.999f;
	
	struct RaytraceDvrOutput
	{
		float3 color; //XYZ-space
		float alpha;
		float3 normalWorld;
		float depth;
	};


	struct DvrUtils
	{
		template<VolumeFilterMode FilterMode, bool Normalize=true>
		static __device__ float3 computeNormal(const float3& volPos, float normalStepSize)
		{
			float3 normal;
			normal.x = 0.5 * (customTex3D(volPos.x + normalStepSize, volPos.y, volPos.z, integral_constant<int, FilterMode>())
				- customTex3D(volPos.x - normalStepSize, volPos.y, volPos.z, integral_constant<int, FilterMode>()));
			normal.y = 0.5 * (customTex3D(volPos.x, volPos.y + normalStepSize, volPos.z, integral_constant<int, FilterMode>())
				- customTex3D(volPos.x, volPos.y - normalStepSize, volPos.z, integral_constant<int, FilterMode>()));
			normal.z = 0.5 * (customTex3D(volPos.x, volPos.y, volPos.z + normalStepSize, integral_constant<int, FilterMode>())
				- customTex3D(volPos.x, volPos.y, volPos.z - normalStepSize, integral_constant<int, FilterMode>()));
			if (Normalize) {
				normal = safeNormalize(normal);
			}
			return normal;
		}

		static __device__ float transformDensity(float density)
		{
			float nval = density;
			if (nval < cDeviceSettings.minDensity) return -1;
			nval = fminf(nval, cDeviceSettings.maxDensity);
			nval = (nval - cDeviceSettings.minDensity) / (cDeviceSettings.maxDensity - cDeviceSettings.minDensity);
			return nval;
		}

		//inverse of transformDensity
		static __device__ float detransformDensity(float nval)
		{
			return nval * (cDeviceSettings.maxDensity - cDeviceSettings.minDensity) + cDeviceSettings.minDensity;
		}

		static __device__ bool evalTf(float density, float3& rgb, float& opacity)
		{
			float nval = transformDensity(density);
			if (nval < 0) return false;
			auto rgba = tex1D<float4>(cDeviceSettings.tfTexture, nval);
			rgb = make_float3(rgba);
			opacity = rgba.w * cDeviceSettings.opacityScaling;
			return true;
		}

		static __device__ float3 phongShading(const float3& colorIn, const float3& normal, const float3& rayDir)
		{
			float3 color = make_float3(0);
			color += cDeviceSettings.shading.ambientLightColor * colorIn; //ambient light
			color += cDeviceSettings.shading.diffuseLightColor * colorIn *
				abs(dot(normal, cDeviceSettings.shading.lightDirection)); //diffuse
			float3 reflect = 2 * dot(cDeviceSettings.shading.lightDirection, normal) *
				normal - cDeviceSettings.shading.lightDirection;
			//int exponent = static_cast<int>(cDeviceSettings.shading.specularExponent);
			float exponent = round(cDeviceSettings.shading.specularExponent);
			color += cDeviceSettings.shading.specularLightColor * (
				((exponent + 2) / (2 * M_PI)) *
				pow(clamp(dot(reflect, rayDir), 0.0f, 1.0f), exponent));
			return color;
		}
	};

	template<VolumeFilterMode FilterMode>
	struct DvrFixedStepSizeTexture
	{
		/**
		 * The raytracing kernel for DVR (Direct Volume Rendering).
		 * The transfer function is specified as 1D texture.
		 *
		 * Input:
		 *  - settings: additional render settings
		 *  - volumeTex: the 3D volume texture
		 *  - tfTexture: transfer function texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Returns RGBA value accumulated through the ray direction.
		 */
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax, 
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			float3 rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			float oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;
			for (real sampleDepth = fmax(0.0f, tmin); sampleDepth < tmax && oBuffer < OpacityEarlyOut; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + float(sampleDepth) * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				KERNEL_INSTRUMENTATION_INC(densityFetches);

				float3 rgb_f;
				float opacity_f;
				KERNEL_INSTRUMENTATION_INC(tfFetches);
				if (!DvrUtils::evalTf(nval, rgb_f, opacity_f))
					continue;

				if (opacity_f > 1e-4)
				{
					opacity_f *= cDeviceSettings.stepsize;
					
					//compute normal
					float3 normal = make_float3(0, 0, 0);
					if (cDeviceSettings.useShading)
					{
						KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
						normal = DvrUtils::computeNormal<FilterMode>(volPos, cDeviceSettings.normalStepSize);
						rgb_f = DvrUtils::phongShading(rgb_f, normal, rayDir);
					}
					float3 rgb = rgb_f;
					opacity_f = rmin(1.0f, opacity_f);
					rgbBuffer += (1.0f - oBuffer) * opacity_f * rgb;
					normalBuffer += (1.0f - oBuffer) * opacity_f * normal;
					depthBuffer += (1.0f - oBuffer) * opacity_f * float(sampleDepth);
					oBuffer += (1.0f - oBuffer) * opacity_f;
				}
			}

			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};

	template<int D>
	__device__ __inline__ void DvrFixedStepSizePreintegrate_Eval(
		float dStart, float dEnd, float stepsize,
		const cudaTextureObject_t& tex,
		float3& rgb_out, float& opacity_out);
	template<>
	__device__ __inline__ void DvrFixedStepSizePreintegrate_Eval<1>(
		float sf, float sb, float stepsize,
		const cudaTextureObject_t& tex,
		float3& rgb_out, float& opacity_out)
	{
		float4 Vsf = tex1D<float4>(tex, sf);
		float4 Vsb = tex1D<float4>(tex, sb);
		opacity_out = 1 - expf(-stepsize * (Vsb.w - Vsf.w) / (sb - sf));
		rgb_out = stepsize * (make_float3(Vsb) - make_float3(Vsf)) / (sb - sf);
		//printf("dStart=%.3f, dEnd=%.3f -> rgb=(%.3f, %.3f, %.3f), alpha=%.3f\n",
		//	sf, sb, rgb_out.x, rgb_out.y, rgb_out.z, opacity_out);
	}
	template<>
	__device__ __inline__ void DvrFixedStepSizePreintegrate_Eval<2>(
		float sf, float sb, float stepsize,
		const cudaTextureObject_t& tex,
		float3& rgb_out, float& opacity_out)
	{
		float4 v = tex2D<float4>(tex, sf, sb);
		rgb_out = make_float3(v);
		opacity_out = v.w;
		//printf("dStart=%.3f, dEnd=%.3f -> rgb=(%.3f, %.3f, %.3f), alpha=%.3f\n",
		//	sf, sb, rgb_out.x, rgb_out.y, rgb_out.z, opacity_out);
	}
	
	template<VolumeFilterMode FilterMode, int D>
	struct DvrFixedStepSizePreintegrate
	{
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			float3 rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			float oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;
			float previousDensity = -1;
			for (real sampleDepth = fmax(0.0f, tmin); sampleDepth < tmax && oBuffer < OpacityEarlyOut; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + float(sampleDepth) * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				KERNEL_INSTRUMENTATION_INC(densityFetches);

				float3 rgb_f;
				float opacity_f = 0;
				KERNEL_INSTRUMENTATION_INC(tfFetches);
				//pre-integration
				if (previousDensity >= 0)
				{
					DvrFixedStepSizePreintegrate_Eval<D>(
						previousDensity, nval, cDeviceSettings.stepsize,
						cDeviceSettings.tfTexture, rgb_f, opacity_f);
				}
				previousDensity = nval;

				if (opacity_f > 1e-4)
				{
					//compute normal
					float3 normal = make_float3(0, 0, 0);
					if (cDeviceSettings.useShading)
					{
						KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
						normal = DvrUtils::computeNormal<FilterMode>(volPos, cDeviceSettings.normalStepSize);
						rgb_f = DvrUtils::phongShading(rgb_f, normal, rayDir);
					}
					float3 rgb = rgb_f;
					opacity_f = rmin(1.0f, opacity_f);
					rgbBuffer += (1.0f - oBuffer) * rgb;
					normalBuffer += (1.0f - oBuffer) * opacity_f * normal;
					depthBuffer += (1.0f - oBuffer) * opacity_f * float(sampleDepth);
					oBuffer += (1.0f - oBuffer) * opacity_f;
				}
			}

			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};
	
	template<VolumeFilterMode FilterMode>
	struct DvrFixedStepSizeControlPoints
	{
		/**
		 * The raytracing kernel for DVR (Direct Volume Rendering).
		 * The transfer function is specified by discrete control points.
		 *
		 * Input:
		 *  - settings: additional render settings
		 *  - volumeTex: the 3D volume texture
		 *  - tfTexture: transfer function texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Returns RGBA value accumulated through the ray direction.
		 */
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			float3 rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			float oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;
			for (real sampleDepth = fmax(0.0f, tmin); sampleDepth < tmax && oBuffer < OpacityEarlyOut; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + float(sampleDepth) * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float tfDensity = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				KERNEL_INSTRUMENTATION_INC(densityFetches);

				KERNEL_INSTRUMENTATION_INC(tfFetches);
				int controlPoint = cDeviceSettings.tfPoints.searchInterval(tfDensity);
				float4 rgbOpacity = cDeviceSettings.tfPoints.queryDvr(tfDensity, controlPoint);
				float3 rgb_f = make_float3(rgbOpacity);
				float opacity_f = rgbOpacity.w * cDeviceSettings.opacityScaling;

				if (opacity_f > 1e-4)
				{
					opacity_f *= cDeviceSettings.stepsize;

					//compute normal
					float3 normal = make_float3(0, 0, 0);
					if (cDeviceSettings.useShading)
					{
						KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
						normal = DvrUtils::computeNormal<FilterMode>(volPos, cDeviceSettings.normalStepSize);
						rgb_f = DvrUtils::phongShading(rgb_f, normal, rayDir);
					}
					float3 rgb = rgb_f;
					opacity_f = rmin(1.0f, opacity_f);
					rgbBuffer += (1.0f - oBuffer) * opacity_f * rgb;
					normalBuffer += (1.0f - oBuffer) * opacity_f * normal;
					depthBuffer += (1.0f - oBuffer) * opacity_f * float(sampleDepth);
					oBuffer += (1.0f - oBuffer) * opacity_f;
				}
			}

			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};

	template<VolumeFilterMode FilterMode>
	struct DvrFixedStepSizeControlPoints_Doubles
	{
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			double3 rgbBuffer = make_double3(0.0f, 0.0f, 0.0f);
			double oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;
			for (double sampleDepth = fmax(0.0f, tmin); sampleDepth < tmax && oBuffer < OpacityEarlyOut; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + float(sampleDepth) * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float tfDensity = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				KERNEL_INSTRUMENTATION_INC(densityFetches);

				KERNEL_INSTRUMENTATION_INC(tfFetches);
				int controlPoint = cDeviceSettings.tfPoints.searchInterval(tfDensity);
				float4 rgbOpacity = cDeviceSettings.tfPoints.queryDvr(tfDensity, controlPoint);
				float3 rgb_f = make_float3(rgbOpacity);
				float opacity_f = rgbOpacity.w * cDeviceSettings.opacityScaling;

				if (opacity_f > 1e-4)
				{
					opacity_f *= cDeviceSettings.stepsize;

					//compute normal
					float3 normal = make_float3(0, 0, 0);
					if (cDeviceSettings.useShading)
					{
						KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
						normal = DvrUtils::computeNormal<FilterMode>(volPos, cDeviceSettings.normalStepSize);
						rgb_f = DvrUtils::phongShading(rgb_f, normal, rayDir);
					}
					float3 rgb = rgb_f;
					opacity_f = rmin(1.0f, opacity_f);
					rgbBuffer += (1.0f - oBuffer) * opacity_f * make_double3(rgb);
					normalBuffer += (1.0f - oBuffer) * opacity_f * normal;
					depthBuffer += (1.0f - oBuffer) * opacity_f * float(sampleDepth);
					oBuffer += (1.0f - oBuffer) * opacity_f;
				}
			}

			return { make_float3(rgbBuffer), float(oBuffer), normalBuffer, depthBuffer };
		}
	};

	template<VolumeFilterMode FilterMode, bool Coupled>
	struct DvrAdaptiveStepSize
	{
		/**
		 * The raytracing kernel for DVR (Direct Volume Rendering).
		 * Using adaptive simpson quadrature based on
		 * Campagnolo 2015, Accurate Volume Rendering based on Adaptive Numerical Integration
		 *
		 * Not working, infinite loop
		 *
		 * Input:
		 *  - settings: additional render settings
		 *  - volumeTex: the 3D volume texture
		 *  - tfTexture: transfer function texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Returns RGBA value accumulated through the ray direction.
		 */
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			tmin = fmax(0.0f, tmin);
			float hInitial = maxCoeff(cDeviceSettings.voxelSize);
			float hMin = hInitial * 1e-3;
			float hMax = fminf(1.0f, hInitial * 1e2f);
			float epsilon = cDeviceSettings.stepsize;

			float3 rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			float oBuffer = 0.0f;
			float sumInnerIntegralValue = 0;
			
			float hInner, hOuter;
			hInner = hInitial;
			if (!Coupled) hOuter = hInner;
			while (tmin < tmax && oBuffer < OpacityEarlyOut)
			{
				//evaluate inner integral
				float currentHInner = hInner, nextHInner, nextEpsilon;
				const auto innerIntegral = [&rayStart, &rayDir, &instrumentation](float t)
				{
					float3 npos = rayStart + t * rayDir;
					float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
					float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
					KERNEL_INSTRUMENTATION_INC(densityFetches);
					float3 rgb_f;
					float opacity_f;
					KERNEL_INSTRUMENTATION_INC(tfFetches);
					DvrUtils::evalTf(nval, rgb_f, opacity_f);
					return opacity_f;
				};
				float innerIntergralValue = Quadrature::adaptiveSimpsonSingleStep
					<decltype(innerIntegral), float, float>(
					innerIntegral, tmin, tmin+hInner, currentHInner, epsilon, nextHInner, nextEpsilon,
					hMin, hMax);
				if (innerIntergralValue != 0)
				{
					sumInnerIntegralValue += innerIntergralValue;
					float T = expf(-sumInnerIntegralValue);
					oBuffer = T;
					if (Coupled) hOuter = hInner;
					//outer integral
					const auto outerIntegral = [T, &rayStart, &rayDir, &instrumentation](float t)
					{
						float3 npos = rayStart + t * rayDir;
						float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
						float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
						KERNEL_INSTRUMENTATION_INC(densityFetches);
						float3 rgb_f;
						float opacity_f;
						KERNEL_INSTRUMENTATION_INC(tfFetches);
						DvrUtils::evalTf(nval, rgb_f, opacity_f);

						//compute normal
						float3 normal = make_float3(0, 0, 0);
						if (cDeviceSettings.useShading)
						{
							KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
							normal = DvrUtils::computeNormal<FilterMode>(volPos, cDeviceSettings.normalStepSize);
							rgb_f = DvrUtils::phongShading(rgb_f, normal, rayDir);
						}
						rgb_f *= opacity_f;
						return rgb_f * T;
					};
					rgbBuffer += Quadrature::adaptiveSimpson
						<decltype(outerIntegral), float, float3>(
						outerIntegral, tmin, tmin + hInner, hOuter, epsilon, hMin, hMax);
					
				}
				
				tmin += hInner;
				hInner = nextHInner; //global step size adaption
				epsilon = nextEpsilon;
			}
			
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;
			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};

	//Reducing Artifacts in Volume Rendering by Higher Order Integration
	//de Boer et al, 1997
	//DEPRECATED
	template<VolumeFilterMode FilterMode>
	struct DvrSimpson1
	{
		/**
		 * The raytracing kernel for DVR (Direct Volume Rendering).
		 *
		 * Input:
		 *  - settings: additional render settings
		 *  - volumeTex: the 3D volume texture
		 *  - tfTexture: transfer function texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Returns RGBA value accumulated through the ray direction.
		 */
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax, 
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			auto rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;
			const float3 volumeResolutionF = make_float3(cDeviceSettings.volumeResolution);

			float4 prevRGBA = make_float4(0);
			float4 currentRGBA = make_float4(0);
			float3 normal;

			for (float sampleDepth = max(0.0f, tmin); sampleDepth < tmax && oBuffer < 0.999f; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + sampleDepth * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());

				float3 rgb;
				float opacity;
				if (!DvrUtils::evalTf(nval, rgb, opacity))
					continue;
				opacity *= cDeviceSettings.stepsize;

				if (opacity > 1e-4)
				{
					//compute normal
					normal = DvrUtils::computeNormal<FilterMode>(volPos, cDeviceSettings.normalStepSize);
					if (cDeviceSettings.useShading)
					{
						rgb = DvrUtils::phongShading(rgb, normal, rayDir);
					}

					currentRGBA = make_float4(rgb, opacity);
				}

				if (prevRGBA.w > 1e-4 || currentRGBA.w > 1e-4)
				{
					const float scaling = 2.0f / 3.0f; //scaling due to Simpsons rule
					float4 interpRgb = 0.5 * (prevRGBA + currentRGBA);
					interpRgb.w = fminf(1.0, interpRgb.w);
					rgbBuffer += (1.0f - oBuffer) * scaling * interpRgb.w * make_float3(interpRgb.x, interpRgb.y, interpRgb.z);
					oBuffer += (1.0f - oBuffer) * scaling * interpRgb.w;
				}
				prevRGBA = currentRGBA;

				if (opacity > 1e-4)
				{
					const float scaling = 1.0f / 3.0f; //scaling due to Simpsons rule
					opacity = fminf(1.0, opacity);
					rgbBuffer += (1.0f - oBuffer) * scaling * opacity * make_float3(rgb.x, rgb.y, rgb.z);
					normalBuffer += (1.0f - oBuffer) * opacity * normal;
					depthBuffer += (1.0f - oBuffer) * opacity * sampleDepth;
					oBuffer += (1.0f - oBuffer) * scaling * opacity;
				}


			}

			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};

	//Evaluators of the isosurface within a voxel
	//The voxel is defined by the eight surrounding vertices
	//The isosurface is already subtracted, only the intersection needs to be found
	struct IDvrVoxelEval
	{
		/**
		 * \brief Evaluates the current voxel, searches for the zero intersection
		 * \param vals the eight corner values xyz=[(0,0,0), (1,0,0), (0,1,0), ..., (1,1,1)]
		 * \param entry the entry point to the voxel in [0,voxelSize]^3
		 * \param dir the direction within the voxel
		 * \param tExit the time of exit
		 * \param timeOut the time of intersection
		 * \param normalOut the normal at the intersection
		 * \param debug use debug prints
		 * \return true iff an intersection was found
		 */
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& volPos,
			const float3& dir, float tCurrent, float tExit,
			float4& rgbDensityOut, float3& normalOut, float& depthOut,
			bool debug, PerPixelInstrumentation* instrumentation);
	};

	struct DvrVoxelEvalFixedStepsTexture : IDvrVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& volPos,
			const float3& dir, float tCurrent, float tExit,
			float4& rgbDensityOut, float3& normalOut, float& depthOut, 
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			//TODO: normal and depth are ignored for now
			const float stepsize = cDeviceSettings.stepsize;
			const float3& voxelSize = cDeviceSettings.voxelSize;
			float3 rgbBuffer = make_float3(0, 0, 0);
			float oBuffer = 0;
			for (float time = 0; time < tExit; time += stepsize)
			{
				float3 p = entry + (time * dir / voxelSize);
				float nval = lerp3D(vals, p);

				float3 rgb;
				float opacity;
				KERNEL_INSTRUMENTATION_INC(tfFetches);
				if (DvrUtils::evalTf(nval, rgb, opacity))
				{
					float opacityR = rmin(1.0f, opacity * rmin(stepsize, tExit - time));
					rgbBuffer += (1.0f - oBuffer) * opacityR * rgb;
					oBuffer += (1.0f - oBuffer) * opacityR;
				}
			}
			rgbDensityOut = make_float4(rgbBuffer, oBuffer);
#ifndef KERNEL_NO_DEBUG
			if (debug && oBuffer > 1e-5) {
				printf("  data found, rgb=(%.4f, %.4f, %.4f), o=%.4f\n",
					rgbBuffer.x, rgbBuffer.y, rgbBuffer.z, oBuffer);
			}
#endif
			return oBuffer > 1e-7;
		}
	};
	struct DvrVoxelEvalFixedStepsControlPoints : IDvrVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& volPos,
			const float3& dir, float tCurrent, float tExit,
			float4& rgbDensityOut, float3& normalOut, float& depthOut, 
			bool debug, PerPixelInstrumentation* instrumentation)
		{	
			//TODO: normal and depth are ignored for now
			const float stepsize = cDeviceSettings.stepsize;
			const float3& voxelSize = cDeviceSettings.voxelSize;
			float3 rgbBuffer = make_float3(0, 0, 0);
			float oBuffer = 0;
			for (float time = 0; time < tExit; time += stepsize)
			{
				float3 p = entry + (time * dir / voxelSize);
				float nval = lerp3D(vals, p);

				//float tfDensity = DvrUtils::transformDensity(settings, nval);
				float tfDensity = nval; //points are already transformed
				if (tfDensity >= 0)
				{
					KERNEL_INSTRUMENTATION_INC(tfFetches);
					int controlPoint = cDeviceSettings.tfPoints.searchInterval(tfDensity);
					float4 rgbOpacity = cDeviceSettings.tfPoints.queryDvr(tfDensity, controlPoint);
					float3 rgb = make_float3(rgbOpacity);
					float opacity = rgbOpacity.w * cDeviceSettings.opacityScaling;

					opacity = rmin(1.0f, opacity * rmin(stepsize, tExit - time));
					rgbBuffer += (1.0f - oBuffer) * opacity * rgb;
					oBuffer += (1.0f - oBuffer) * opacity;
				}
			}
			rgbDensityOut = make_float4(rgbBuffer, oBuffer);
#ifndef KERNEL_NO_DEBUG
			if (debug && oBuffer > 1e-5) {
				printf("  data found, rgb=(%.4f, %.4f, %.4f), o=%.4f\n",
					rgbBuffer.x, rgbBuffer.y, rgbBuffer.z, oBuffer);
			}
#endif
			return oBuffer > 1e-5;
		}
	};




	//============================================================================
	// MAIN RAY-SPLITTING ALGORITHM
	//============================================================================

	/**
	 * \brief Ray-Splitting Algorithm.
	 * Uses the curiously recurring template pattern
	 * to select implementations for EmitInterval and EmitPoint
	 */
	template<
		typename Derived,
		int MarmittNumIterations = 3,
		bool ExtendedEarlyOut = false>
	struct RaySplittingAlgorithm : IDvrVoxelEval
	{
		/**
		 * \brief Computes the interval
		 *  \f$ \int_{tEntry}^{tExit} L(x) exp(O(x)) dx $\f
		 * where L is the emission and O the absorption.
		 * The mapping from x to density is given by the cubic polynomial 'poly'.
		 * The transfer function is linear in the density, given by
		 *    dataEntry = (L(tf(tEntry)), O(tf(tEntry)))
		 * and
		 *    dataExit = (L(tf(tExit)), O(tf(tExit))) .
		 *
		 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
		 *
		 * <b> Must be overwritten from subclasses if used</b>
		 * 
		 * \param poly
		 * \param tEntry
		 * \param tExit
		 * \param dataEntry
		 * \param dataExit
		 * \param rgbBufferOut the blended rgb color (premultiplied with alpha)
		 * \param oBufferOut the blended opacity / alpha
		 */
		static __host__ __device__ __inline__ void emitInterval(
			const float vals[8], const float4& poly, float tEntry, float tExit,
			const float3& volPos, const float3& rayDir,
			float densityEntry, float densityExit,
			const float4& dataEntry, const float4& dataExit,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
		}

		/**
		 * \brief Evaluates the contribution at the current control point.
		 * Used for isosurface renderings.
		 *
		 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
		 *
		 * <b> Must be overwritten from subclasses if used</b>
		 *
		 * \param volumeTexLinear the texture for normal computation
		 * \param volPos the current position in the volume
		 * \param rayDir the direction of the ray
		 * \param sampleDepth the current sample depth
		 * \param data the color+absorption at the current point
		 * \param rgbBufferOut the blended rgb color (premultiplied with alpha)
		 * \param oBufferOut the blended opacity / alpha
		 */
		static __device__ __inline__ void emitPoint(
			const float3& volPos, const float3& rayDir,
			float sampleDepth, const float4& data,
			float3& rgbBufferOut, float3& normalBufferOut,
			float& depthBufferOut, float& oBufferOut,
			PerPixelInstrumentation* instrumentation)
		{
		}
		
		
		//main ray splitting algorithm
		static __host__ __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& volPos,
			const float3& dir, float tCurrent, float tExit,
			float4& rgbDensityOut, float3& normalOut, float& depthOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
#ifndef KERNEL_NO_DEBUG
			int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; //debug
#endif

			//early out
			if (tExit < 1e-7)
				return false; //not much to do here
			const auto maxDensity = kernel::maximum(
				vals[0], vals[1], vals[2], vals[3],
				vals[4], vals[5], vals[6], vals[7]);
			if (maxDensity < cDeviceSettings.realMinDensity)
				return false;

			const float3 dir2 = dir / cDeviceSettings.voxelSize;

			//normal and depth are only used for isosurfaces (EmitPoint) for now
			float3 rgbBuffer = make_float3(0, 0, 0);
			float oBuffer = 0;
			float3 normalBuffer = make_float3(0, 0, 0);
			float depthBuffer = 0;

			//find initial control point interval
			float timeEntry = 0;
			float densityEntry = lerp3D(vals, entry);
			KERNEL_INSTRUMENTATION_INC(tfFetches);
			int currentIndex = cDeviceSettings.tfPoints.searchInterval(densityEntry);
			float4 dataEntry = cDeviceSettings.tfPoints.queryDvr(densityEntry, currentIndex);

			//early out if the exit point is in the same interval
			// and the opacity in that interval is always zero.
			//This is disabled by default, as it does not improve the performance,
			// even decrease it slightly.
			if constexpr (ExtendedEarlyOut) {
				if (dataEntry.w == 0) {
					//if the current alpha is zero, the whole interval
					//is very likely to be zero everywhere
					const float pLow = cDeviceSettings.tfPoints.positions[currentIndex];
					const float pHigh = cDeviceSettings.tfPoints.positions[currentIndex + 1];
					const auto minDensity = kernel::minimum(
						vals[0], vals[1], vals[2], vals[3],
						vals[4], vals[5], vals[6], vals[7]);
					if (pLow < minDensity && maxDensity < pHigh)
						return false; //early out, always in a zero-interval
				}
			}

			float intersections[TF_MAX_CONTROL_POINTS][4];
			int currentIntersection[TF_MAX_CONTROL_POINTS] = { 0 };
			//memset(intersections, 0x7f, sizeof(float)* TF_MAX_CONTROL_POINTS * 4); //fill with a large value (3.39615e+38)

			//compute all intersections
			const float4 poly = CubicPolynomial<float>::getFactors(vals, entry, dir2);
			intersections[0][0] = FLT_MAX;
			for (int i = 1; i < cDeviceSettings.tfPoints.numPoints - 1; ++i) //we can skip the first and last control point as they are out-of-bounds
			{
				int numIntersections = Marmitt<MarmittNumIterations>::evalAll(
					vals, poly, cDeviceSettings.tfPoints.positions[i],
					entry, dir2, timeEntry, tExit,
					intersections[i]);
#ifndef KERNEL_NO_DEBUG
				if (cDeviceSettings.voxelFiltered && debug)
				{
					printf("[%05d] iso %d, num=%d, pos={%.4f, %.4f, %.4f}\n",
						globalThreadIdx, i, numIntersections,
						numIntersections > 0 ? intersections[i][0] : -1.0f,
						numIntersections > 1 ? intersections[i][1] : -1.0f,
						numIntersections > 2 ? intersections[i][2] : -1.0f);
				}
#endif
				intersections[i][numIntersections] = FLT_MAX; //end-marker
			}
			intersections[cDeviceSettings.tfPoints.numPoints - 1][0] = FLT_MAX;
			KERNEL_INSTRUMENTATION_ADD(isoIntersections, cDeviceSettings.tfPoints.numPoints - 2);

			//DEBUG: massive logging if voxel is selected
#ifndef KERNEL_NO_DEBUG
			if (cDeviceSettings.voxelFiltered && debug)
			{
				printf("[%05d] vals={%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f}, entry={%.4f, %.4f, %.4f}, dir={%.4f, %.4f, %.4f}, tExit=%.4f\n",
					globalThreadIdx, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7],
					entry.x, entry.y, entry.z, dir2.x, dir2.y, dir2.z, tExit);
				printf("[%05d] entry density=%.4f, index=%d\n", globalThreadIdx, densityEntry, currentIndex);
				for (int i = 1; i < cDeviceSettings.tfPoints.numPoints - 1; ++i)
					printf("[%05d] control point %d, density=%.4f, intersections={%.4f, %.4f, %.4f}\n",
						globalThreadIdx, i, cDeviceSettings.tfPoints.positions[i],
						intersections[i][0] < FLT_MAX ? intersections[i][0] : -1.f,
						intersections[i][1] < FLT_MAX ? intersections[i][1] : -1.f,
						intersections[i][2] < FLT_MAX ? intersections[i][2] : -1.f);
			}
#endif

			int indexPrevious = -1;
			while (true) // breaks if no more intersections are found
			{
				//invariants: since we don't neccessarily have points at d=0,
				//we can run out of points to check
				if (currentIndex < 0 || currentIndex + 1 >= cDeviceSettings.tfPoints.numPoints)
					break;

				float timeLower = intersections[currentIndex][currentIntersection[currentIndex]];
				float timeUpper = intersections[currentIndex + 1][currentIntersection[currentIndex + 1]];
#ifndef KERNEL_NO_DEBUG
				if (debug)
				{
					printf("[%05d] current idx=%d, tEntry=%.4f, timeLower=%.4f, timeUpper=%.4f, tExit=%.4f\n",
						globalThreadIdx, currentIndex, timeEntry, timeLower, timeUpper, tExit);
				}
#endif
				if (tExit < timeLower && tExit < timeUpper)
				{
					//exit the voxel first
					const float densityExit = lerp3D(vals, entry + tExit * dir2);
					KERNEL_INSTRUMENTATION_INC(tfFetches);
					const int controlPoint = cDeviceSettings.tfPoints.searchInterval(densityExit);
					const float4 dataExit = cDeviceSettings.tfPoints.queryDvr(densityExit, controlPoint);
#ifndef KERNEL_NO_DEBUG
					if (debug)
					{
						printf("[%05d] exit! density=[%.4f, %.4f], time=[%.4f, %.4f]\n",
							globalThreadIdx, densityEntry, densityExit, timeEntry, tExit);
					}
#endif
					Derived::emitInterval(
						vals, poly, timeEntry, tExit,
						entry, dir2,
						densityEntry, densityExit,
						dataEntry, dataExit,
						rgbBuffer, oBuffer,
						debug, instrumentation);
					break;
				}
				else if (timeLower < timeUpper)
				{
					//lower poly
					float4 dataExit;
					float densityExit;
					if (currentIndex == indexPrevious)
					{
						densityExit = cDeviceSettings.tfPoints.positions[currentIndex + 1];
						dataExit = cDeviceSettings.tfPoints.valuesDvr[currentIndex + 1];
#ifndef KERNEL_NO_DEBUG
						if (debug)
						{
							printf("[%05d] case IIIa, density=[%.4f, %.4f], time=[%.4f, %.4f]\n",
								globalThreadIdx, densityEntry, densityExit, timeEntry, timeLower);
						}
#endif
					}
					else
					{
						densityExit = cDeviceSettings.tfPoints.positions[currentIndex];
						dataExit = cDeviceSettings.tfPoints.valuesDvr[currentIndex];
#ifndef KERNEL_NO_DEBUG
						if (debug)
						{
							printf("[%05d] case II, density=[%.4f, %.4f], time=[%.4f, %.4f]\n",
								globalThreadIdx, densityEntry, densityExit, timeEntry, timeLower);
						}
#endif
					}
					KERNEL_INSTRUMENTATION_INC(tfFetches);
					Derived::emitInterval(
						vals, poly, timeEntry, timeLower,
						entry, dir2,
						densityEntry, densityExit,
						dataEntry, dataExit,
						rgbBuffer, oBuffer,
						debug, instrumentation);
					Derived::emitPoint(
						volPos + dir * timeLower, dir, tCurrent + timeLower,
						cDeviceSettings.tfPoints.valuesIso[currentIndex],
						rgbBuffer, normalBuffer, depthBuffer, oBuffer,
						instrumentation);
					densityEntry = densityExit;
					dataEntry = cDeviceSettings.tfPoints.valuesDvr[currentIndex];
					indexPrevious = currentIndex;
					timeEntry = timeLower;
					currentIntersection[currentIndex]++; //pop the intersection
					currentIndex--;
				}
				else
				{
					//upper poly
					float4 dataExit;
					float densityExit;
					if (currentIndex + 1 == indexPrevious)
					{
						densityExit = cDeviceSettings.tfPoints.positions[currentIndex];
						dataExit = cDeviceSettings.tfPoints.valuesDvr[currentIndex];
#ifndef KERNEL_NO_DEBUG
						if (debug)
						{
							printf("[%05d] case IIIb, density=[%.4f, %.4f], time=[%.4f, %.4f]\n",
								globalThreadIdx, densityEntry, densityExit, timeEntry, timeUpper);
						}
#endif
					}
					else
					{
						densityExit = cDeviceSettings.tfPoints.positions[currentIndex + 1];
						dataExit = cDeviceSettings.tfPoints.valuesDvr[currentIndex + 1];
#ifndef KERNEL_NO_DEBUG
						if (debug)
						{
							printf("[%05d] case I, density=[%.4f, %.4f], time=[%.4f, %.4f]\n",
								globalThreadIdx, densityEntry, densityExit, timeEntry, timeUpper);
						}
#endif
					}
					KERNEL_INSTRUMENTATION_INC(tfFetches);
					Derived::emitInterval(
						vals, poly, timeEntry, timeUpper,
						entry, dir2,
						densityEntry, densityExit,
						dataEntry, dataExit,
						rgbBuffer, oBuffer,
						debug, instrumentation);
					Derived::emitPoint(
						volPos + dir * timeUpper, dir, tCurrent + timeUpper,
						cDeviceSettings.tfPoints.valuesIso[currentIndex+1],
						rgbBuffer, normalBuffer, depthBuffer, oBuffer,
						instrumentation);
					densityEntry = densityExit;
					dataEntry = cDeviceSettings.tfPoints.valuesDvr[currentIndex + 1];
					indexPrevious = currentIndex + 1;
					timeEntry = timeUpper;
					currentIntersection[currentIndex + 1]++; //pop the intersection
					currentIndex++;
				}
			}

			rgbDensityOut = make_float4(rgbBuffer.x, rgbBuffer.y, rgbBuffer.z, oBuffer);
			normalOut = normalBuffer;
			depthOut = depthBuffer;
#ifndef KERNEL_NO_DEBUG
			if (debug/* && oBuffer > 1e-5*/) {
				printf("  data found, rgb=(%.4f, %.4f, %.4f), o=%.4f\n",
					rgbBuffer.x, rgbBuffer.y, rgbBuffer.z, oBuffer);
			}
			if (debug && cDeviceSettings.voxelFiltered && (globalThreadIdx == 3453))
			{
				printf("[%05d] data found, rgb=(%.4f, %.4f, %.4f), o=%.4f\n",
					globalThreadIdx, rgbBuffer.x, rgbBuffer.y, rgbBuffer.z, oBuffer);
			}
#endif
			return oBuffer > 1e-7;
		}
	};


	//============================================================================
	// END RAY-SPLITTING ALGORITHM
	//============================================================================
	

	struct IDvrIntervalEvaluator
	{
		static __host__ __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir, //position and direction in the volume
			float3& rgbBufferOut, //pre-multiplied with opacity
			float& oBufferOut,
			PerPixelInstrumentation* instrumentation);
	};
	struct ExtractColor
	{
		__host__ __device__ __inline__ float3 operator()(const float4& d) const { return make_float3(d); }
	};
	struct ExtractAbsorption
	{
		__host__ __device__ __inline__ float operator()(const float4& d) const { return d.w; }
	};

	struct DvrIntervalEvaluatorSimple : IDvrIntervalEvaluator
	{
		static __host__ __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			float h = tExit - tEntry;
			float4 dataMid = dataPoly((tEntry + tExit) / 2);
			float3 rgb = make_float3(dataMid) * dataMid.w * h * cDeviceSettings.opacityScaling;
			float opacity = h * dataMid.w * cDeviceSettings.opacityScaling;

			opacity = 1 - expf(-opacity);//fminf(1.0f, opacity);
			//rgb = fminf(rgb, make_float3(1.0f));
			rgbBufferOut = rgb;
			oBufferOut = opacity;

			KERNEL_INSTRUMENTATION_INC(intervalEval);
			KERNEL_INSTRUMENTATION_INC(intervalStep);
		}
	};

	/**
	 * \brief Evaluates the per-segment integral via stepping (Rectangular rule)
	 * \tparam N the number of steps or -1 for dynamic count
	 */
	template<int N>
	struct DvrIntervalEvaluatorStepping : IDvrIntervalEvaluator
	{
		static __host__ __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			//first, extract the polynomials
			auto absorption = dataPoly.cast(ExtractAbsorption()) * cDeviceSettings.opacityScaling;
			auto color = dataPoly.cast(ExtractColor());
			//auto emission = absorption * color; //CUDA can't deduce that :(
			auto emission = mul<3, 3, float3, float>(color, absorption);
			KERNEL_INSTRUMENTATION_INC(intervalEval);
			
			//now do the stepping
			static const float referenceH = 1e-3;
			int myN;
			if (N < 0)
				myN = int(ceil((tExit - tEntry) / referenceH));
			else
				myN = N;
			float h = (tExit - tEntry) / myN;
			for (int i = 0; i < myN; ++i)
			{
				float t = tEntry + h * i;
				float3 emissionTimesOpacity = emission(make_float3(t)) * h;
				//float absorpt = fminf(1.0f, absorption(t) * h);
				float absorpt = 1 - expf(-absorption(t) * h);

				//blend it (this stays the same)
				rgbBufferOut += (1.0f - oBufferOut) * emissionTimesOpacity;
				oBufferOut += (1.0f - oBufferOut) * absorpt;

				KERNEL_INSTRUMENTATION_INC(intervalStep);
			}
		}
	};

	template<int N>
	struct DvrIntervalEvaluatorTrapezoid_ErrorEstimate
	{
		template<typename PolyExpPoly_t>
		static __host__ __device__ __inline__ CONSTEXPR
		int evalN(const PolyExpPoly_t& polyExpPoly, float a, float b, float allowedError)
		{
			static_assert(N > 0, "N must be > 0 for the static case");
			return N; //base case
		}
	};
	template<>
	struct DvrIntervalEvaluatorTrapezoid_ErrorEstimate<-1> //dynamic case
	{
		template<typename PolyExpPoly_t>
		static __host__ __device__ __inline__
		int evalN(const PolyExpPoly_t& polyExpPoly, float a, float b, float allowedError)
		{
			//compute second derivative and expand
			const auto d2 = polyExpPoly.d2().expand();
			//compute upper bound
			const float3 beta3 = d2.absBoundBernstein(a, b);
			const float beta = fmaxf(beta3.x, fmaxf(beta3.y, beta3.z));
			//compute N
			const float hPrime = sqrtf(12 * allowedError / ((b - a) * beta));
			int N = static_cast<int>(ceilf((b - a) / hPrime));
			static const int MAX_N = 100; //arbitrary upper bound
			N = max(2, min(N, MAX_N));
			return N;
		}
	};
	
	/**
	 * \brief Evaluates the per-segment integral via the trapezoid rule.
	 * 
	 * If N is -1 then the number of steps is determined by bounding the error.
	 * The current stepsize determines the maximal error and an error estimate is
	 * used to select the number of steps so that the actual error does not
	 * become larger than that number.
	 * Use instrumentation to query the average and maximal number of steps.
	 * 
	 * \tparam N the number of steps or -1 for dynamic count
	 */
	template<int N>
	struct DvrIntervalEvaluatorTrapezoid : IDvrIntervalEvaluator
	{
		static __host__ __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
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
			const auto fullEmission = 
				polyExpPoly<6, float3, 4, float>(emission, absorptionInt);

			//evaluate using trapezoid rule
			float allowedError = cDeviceSettings.stepsize;
			//int N = max(2.0f, ceilf((tExit - tEntry) / hPrime));
			const int actualN = DvrIntervalEvaluatorTrapezoid_ErrorEstimate<N>::evalN(
				fullEmission, tEntry, tExit, allowedError);
			rgbBufferOut = fullEmission.integrateTrapezoid(tEntry, tExit, actualN);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, N+1);
#if KERNEL_INSTRUMENTATION==1
			if (N > 0) { //log the theoretical needed steps
				int theoreticalN = DvrIntervalEvaluatorTrapezoid_ErrorEstimate<-1>::evalN(
					fullEmission, tEntry, tExit, allowedError);
				KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, theoreticalN + 1);
			} 
			else
				KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, N + 1);
#endif
		}
	};

	template<int N>
	struct DvrIntervalEvaluatorSimpson_ErrorEstimate
	{
		template<typename PolyExpPoly_t>
		static __host__ __device__ __inline__ CONSTEXPR
			int evalN(const PolyExpPoly_t& polyExpPoly, float a, float b, float allowedError)
		{
			static_assert(N > 0, "N must be > 0 for the static case");
			static_assert(N % 2 == 0, "N must be even");
			return N; //base case
		}
	};
	template<>
	struct DvrIntervalEvaluatorSimpson_ErrorEstimate<-1> //dynamic case
	{
		template<typename PolyExpPoly_t>
		static __host__ __device__ __inline__
			int evalN(const PolyExpPoly_t& polyExpPoly, float a, float b, float allowedError)
		{
			//compute forth derivative and expand
			const auto d4 = polyExpPoly.d2().expand().d2().expand();
			//compute upper bound
			const float3 beta3 = d4.absBoundBernstein(a, b);
			const float beta = fmaxf(beta3.x, fmaxf(beta3.y, beta3.z));
			//compute N
			const float hPrime4 = 180 * allowedError / ((b - a) * beta);
			const float hPrime = pow(hPrime4, 1.0f / 4.0f);
			int N = static_cast<int>(ceilf((b - a) / hPrime));
			if (N & 1) ++N; //round up to even
			static const int MAX_N = 100; //arbitrary upper bound
			N = max(2, min(N, MAX_N)); 
			return N;
		}
	};
	
	/**
	 * \brief Evaluates the per-segment integral via the Simpson rule.
	 *
	 * If N is -1 then the number of steps is determined by bounding the error.
	 * The current stepsize determines the maximal error and an error estimate is
	 * used to select the number of steps so that the actual error does not
	 * become larger than that number.
	 * Use instrumentation to query the average and maximal number of steps.
	 *
	 * \tparam N the number of steps or -1 for dynamic count
	 */
	template<int N>
	struct DvrIntervalEvaluatorSimpson : IDvrIntervalEvaluator
	{
		static __host__ __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
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
			const auto fullEmission =
				polyExpPoly<6, float3, 4, float>(emission, absorptionInt);

			//evaluate using Simpson's rule
			float allowedError = cDeviceSettings.stepsize;
			//int N = max(2.0f, ceilf((tExit - tEntry) / hPrime));
			const int actualN = DvrIntervalEvaluatorSimpson_ErrorEstimate<N>::evalN(
				fullEmission, tEntry, tExit, allowedError);
			rgbBufferOut = fullEmission.integrateSimpson(tEntry, tExit, actualN);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, N + 1);
#if KERNEL_INSTRUMENTATION==1
			if (N > 0) { //log the theoretical needed steps
				int theoreticalN = DvrIntervalEvaluatorSimpson_ErrorEstimate<-1>::evalN(
					fullEmission, tEntry, tExit, allowedError);
				KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, theoreticalN + 1);
			}
			else
				KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, N + 1);
#endif
		}
	};

	struct DvrIntervalEvaluatorAdaptiveSimpson : IDvrIntervalEvaluator
	{
		static __host__ __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
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
			const auto fullEmission =
				polyExpPoly<6, float3, 4, float>(emission, absorptionInt);

			//evaluate using Simpson's rule
			//stepsize = user-stepsize * voxelsize, this converts it back to user-stepsize
			float allowedError = cDeviceSettings.stepsize * maxCoeff(make_float3(cDeviceSettings.volumeResolution));;
			float hMax = (tExit - tEntry);
			float hMin = (tExit - tEntry) * 1e-2;
			int N = 0;
			rgbBufferOut = Quadrature::adaptiveSimpson(fullEmission,
				tEntry, tExit, hMax, allowedError, N, hMin);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, N);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, N);

		}
	};

	struct DvrIntervalEvaluatorAdaptiveSimpsonWithShading : IDvrIntervalEvaluator
	{
		static __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			//first, extract the polynomials
			const auto absorption = dataPoly.cast(ExtractAbsorption()) * cDeviceSettings.opacityScaling;
			const auto color = dataPoly.cast(ExtractColor());
			//emission = absorption * color;
			KERNEL_INSTRUMENTATION_INC(intervalEval);

			//absorption can be solved analytically
			auto absorptionInt = absorption.integrate();
			const float absorption0 = absorptionInt(tEntry);
			absorptionInt.coeff[0] -= absorption0;
			absorptionInt = -absorptionInt;
			//now absorptionInt(t1) = -[absorption(t)]_t0^t1
			const float transparency = expf(absorptionInt(tExit));
			oBufferOut = 1 - transparency;

			//build function for full emission ( = emission(x) * exp(absorptionInt(x)) )
			const auto fullEmission = 
				[&entryPos, &rayDir, &instrumentation, &vals, &absorption, &absorptionInt, &color]
				(float time) -> float3
			{
				float currentAbsorption = absorption(time);
				float3 currentColor = color(time);
				float currentIntegratedAbsorption = absorptionInt(time);
				float3 emission = currentColor * currentAbsorption;

				if (currentAbsorption > 1e-4 && cDeviceSettings.useShading)
				{
					//compute phong shading
					float3 volPos1 = entryPos + time * rayDir;
					float3 normal = safeNormalize(lerp3DDerivatives(vals, volPos1));
					emission = DvrUtils::phongShading(emission, normal, rayDir);
				}
				return emission * expf(currentIntegratedAbsorption);
			};

			//evaluate using Simpson's rule
			float allowedError = cDeviceSettings.stepsize;
			float hMax = (tExit - tEntry);
			float hMin = (tExit - tEntry) * 1e-2;
			int N = 0;
			rgbBufferOut = Quadrature::adaptiveSimpson<decltype(fullEmission), float, float3>(
				fullEmission, tEntry, tExit, hMax, allowedError, N, hMin);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, N);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, N);

		}
	};
	template<int N>
	struct DvrIntervalEvaluatorSimpsonWithShading : IDvrIntervalEvaluator
	{
		static __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			//first, extract the polynomials
			const auto absorption = dataPoly.cast(ExtractAbsorption()) * cDeviceSettings.opacityScaling;
			const auto color = dataPoly.cast(ExtractColor());
			//emission = absorption * color;
			KERNEL_INSTRUMENTATION_INC(intervalEval);

			//absorption can be solved analytically
			auto absorptionInt = absorption.integrate();
			const float absorption0 = absorptionInt(tEntry);
			absorptionInt.coeff[0] -= absorption0;
			absorptionInt = -absorptionInt;
			//now absorptionInt(t1) = -[absorption(t)]_t0^t1
			const float transparency = expf(absorptionInt(tExit));
			oBufferOut = 1 - transparency;

			//build function for full emission ( = emission(x) * exp(absorptionInt(x)) )
			const auto fullEmission =
				[&entryPos, &rayDir, &instrumentation, &vals, &absorption, &absorptionInt, &color]
			(float time) -> float3
			{
				float currentAbsorption = absorption(time);
				float3 currentColor = color(time);
				float currentIntegratedAbsorption = absorptionInt(time);
				float3 emission = currentColor * currentAbsorption;

				if (currentAbsorption > 1e-4 && cDeviceSettings.useShading)
				{
					//compute phong shading
					float3 volPos1 = entryPos + time * rayDir;
					float3 normal = safeNormalize(lerp3DDerivatives(vals, volPos1));
					emission = DvrUtils::phongShading(emission, normal, rayDir);
				}
				return emission * expf(currentIntegratedAbsorption);
			};

			//evaluate using Simpson's rule
			rgbBufferOut = Quadrature::simpson<decltype(fullEmission), float, float3>(
				fullEmission, tEntry, tExit, N);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, N);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, N);

		}
	};
	
	/**
	 * \brief Evaluates the per-segment integral via the Simpson rule.
	 *
	 * If N is -1 then the number of steps is determined by bounding the error.
	 * The current stepsize determines the maximal error and an error estimate is
	 * used to select the number of steps so that the actual error does not
	 * become larger than that number.
	 * Use instrumentation to query the average and maximal number of steps.
	 *
	 * \tparam N the number of steps or -1 for dynamic count
	 */
	template<int N>
	struct DvrIntervalEvaluatorPowerSeries : IDvrIntervalEvaluator
	{
		static __host__ __device__ __inline__ void call(
			const float vals[8], const Polynomial<3, float4>& dataPoly, float tEntry, float tExit,
			const float3& entryPos, const float3& rayDir,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
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
			const auto fullEmission =
				polyExpPoly<6, float3, 4, float>(emission, absorptionInt);

			//evaluate using the power series
			rgbBufferOut = fullEmission.integratePowerSeries<N>(tEntry, tExit);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, N + 1);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, N + 1);
		}
	};

#ifndef FLT_MAX //from float.h
#define FLT_MAX          3.402823466e+38F
#endif
	/**
	 * \brief This functor splits the ray inside the voxel into segments
	 * between the TF control points.
	 * Within each segment, the opacity and color are linear functions.
	 * \tparam IntervalEvaluator a subclass of IDvrIntervalEvaluator
	 * \tparam MarmittNumIterations the number of iterations in Marmitt root finding
	 * \tparam ExtendedEarlyOut enables an extra early-out test if the whole
	 *     voxel is in the same zero interval
	 */
	template<typename IntervalEvaluator,
	         int MarmittNumIterations = 3,
	         bool ExtendedEarlyOut = VOXEL_EVAL_INTERVAL_EXTENDED_EARLY_OUT>
	struct DvrVoxelEvalInterval
		: RaySplittingAlgorithm<
				DvrVoxelEvalInterval<IntervalEvaluator, MarmittNumIterations, ExtendedEarlyOut>,
				MarmittNumIterations, ExtendedEarlyOut>
	{
		/**
		 * \brief Computes the interval
		 *  \f$ \int_{tEntry}^{tExit} L(x) exp(O(x)) dx $\f
		 * where L is the emission and O the absorption.
		 * The mapping from x to density is given by the cubic polynomial 'poly'.
		 * The transfer function is linear in the density, given by
		 *    dataEntry = (L(tf(tEntry)), O(tf(tEntry)))
		 * and
		 *    dataExit = (L(tf(tExit)), O(tf(tExit))) .
		 *
		 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
		 * \param poly 
		 * \param tEntry 
		 * \param tExit 
		 * \param dataEntry 
		 * \param dataExit 
		 * \param rgbBufferOut the blended rgb color (premultiplied with alpha)
		 * \param oBufferOut the blended opacity / alpha
		 */
		static __host__ __device__ __inline__ void emitInterval(
			const float vals[8], const float4& poly, float tEntry, float tExit,
			const float3& volPos, const float3& rayDir,
			float densityEntry, float densityExit,
			const float4& dataEntry, const float4& dataExit,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (tExit - tEntry < 1e-6) return; //too small, probably divide-by-zero in lerp
			if (dataEntry.w < 1e-6 && dataExit.w < 1e-6) return; //early out
			
			float3 rgb = { 0,0,0 };
			float opacity = 0;

			auto density = kernel::float4ToPoly<float>(poly);
			auto dataPoly = density.lerp(
				densityEntry, dataEntry,
				densityExit, dataExit);

			//call interval evaluator
			IntervalEvaluator::call(
				vals, dataPoly, tEntry, tExit, volPos, rayDir, 
				rgb, opacity, debug, instrumentation);

#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf(" emit density={%.4f, %.4f, %.4f, %.4f}, opacity={%.4f, %.4f, %.4f, %.4f}\n  -> r=%.4f, g=%.4f, b=%.4f, a=%.4f\n",
					density[0], density[1], density[2], density[3],
					dataPoly[0].w, dataPoly[1].w, dataPoly[2].w, dataPoly[3].w,
					rgb.x, rgb.y, rgb.z, opacity);
			}
#endif
			
			//blend it (this stays the same)
			rgbBufferOut += (1.0f - oBufferOut) * rgb;
			oBufferOut += (1.0f - oBufferOut) * opacity;
		}
	};



	/**
	 * DVR evaluation kernel using voxel traversal.
	 *
	 * The VoxelEvaluation functor is called once each voxel.
	 * It must provide static method
	 *     bool call(vals, entry, rayDir, tExit, settings, tfTexture,
	 *				 rgbDensityOut, normalOut, depthOut);
	 * That returns true if rgbDenstiyOut, normalOut, depthOut are filled
	 * with values that should be blended.
	 * The colors are pre-multiplied with the density
	 */
	template<typename VoxelEvaluation, bool SmoothNormals = true>
	struct DvrDDA
	{

		/**
		 * The raytracing kernel for DVR using DDA stepping.
		 * Requires a nearest-neighbor sampled volume
		 *
		 * Input:
		 *  - settings: additional render settings
		 *  - volumeTex: the 3D volume texture
		 *  - tfTexture: transfer function texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Returns RGBA value accumulated through the ray direction.
		 */
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax, 
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			float3 rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			float oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;

			//from fixed-step ray-casting as reference:
			//float3 npos = rayStart + sampleDepth * rayDir;
			//float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);

			const float3 entryPos = (rayStart + rayDir * (tmin + 1e-7) - cDeviceSettings.boxMin);
			DDA dda(entryPos, rayDir, cDeviceSettings.voxelSize);
			int3 voxelPos = dda.p;

			//it might happen that we start a bit outside due to numerics
			//perform up to three steps in advance and see if we are still outside
			for (int iter = 0;
				iter < 3 &&
				(voxelPos.x < 0 || voxelPos.y < 0 || voxelPos.z < 0 ||
					voxelPos.x >= cDeviceSettings.volumeResolution.x - 1 || voxelPos.y >= cDeviceSettings.volumeResolution.y - 1 || voxelPos.z >= cDeviceSettings.volumeResolution.z - 1);
				iter++)
			{
				dda.step();
				KERNEL_INSTRUMENTATION_INC(ddaSteps);
				voxelPos = dda.p;
			}

			constexpr int maxIter = 2 * (1 << 10);
			int iter;
			for (iter = 0;
#ifndef KERNEL_NO_DEBUG
				iter < maxIter &&
#endif
				voxelPos.x >= 0 && voxelPos.y >= 0 && voxelPos.z >= 0 &&
				voxelPos.x < cDeviceSettings.volumeResolution.x - 1 && voxelPos.y < cDeviceSettings.volumeResolution.y - 1 && voxelPos.z < cDeviceSettings.volumeResolution.z - 1 &&
				oBuffer < OpacityEarlyOut;
				iter++)
			{
				//fetch eight voxels
				float vals[8];
				vals[0] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y, voxelPos.z);
				vals[1] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y, voxelPos.z);
				vals[2] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y + 1, voxelPos.z);
				vals[3] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y + 1, voxelPos.z);
				vals[4] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y, voxelPos.z + 1);
				vals[5] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y, voxelPos.z + 1);
				vals[6] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y + 1, voxelPos.z + 1);
				vals[7] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y + 1, voxelPos.z + 1);
				KERNEL_INSTRUMENTATION_ADD(densityFetches, 8);

				//save current pos and time
				const float3 volPos = entryPos + rayDir * dda.t;
				const float3 entry = volPos / cDeviceSettings.voxelSize - make_float3(voxelPos);
				const float tCurrent = dda.t;

				//step into next voxel
				dda.step();
				KERNEL_INSTRUMENTATION_INC(ddaSteps);

				bool breakOut = false;
				if (dda.t > tmax - tmin)
				{
					breakOut = true;
					dda.t = tmax - tmin; //finish current voxel, then break
				}

				//call voxel evaluator
				const float tExit = dda.t - tCurrent;
				float4 rgbDensityOut;
				float3 normalOut;
				float depthOut;
//#ifndef KERNEL_NO_DEBUG
//				if (debug) {
//					printf(" Visit voxel (%d, %d, %d), entry=(%.4f, %.4f, %.4f), tExit=%.4f, tSide=(%f,%f,%f), tDel=(%.3f,%.3f,%.3f)\n",
//						voxelPos.x, voxelPos.y, voxelPos.z,
//						entry.x, entry.y, entry.z, tExit,
//						dda.tSide.x, dda.tSide.y, dda.tSide.z,
//						dda.tDel.x, dda.tDel.y, dda.tDel.z);
//				}
//#endif
				bool hasData; 
#ifndef KERNEL_NO_DEBUG
				if (cDeviceSettings.voxelFiltered && any(cDeviceSettings.selectedVoxel != voxelPos))
					hasData = false; //filter voxel
				else
					hasData = VoxelEvaluation::call(
						vals, entry, volPos, rayDir, tCurrent, tExit,
						rgbDensityOut, normalOut, depthOut, debug, instrumentation);
#else
				hasData = VoxelEvaluation::call(
					vals, entry, volPos, rayDir, tCurrent, tExit,
					rgbDensityOut, normalOut, depthOut, debug, instrumentation);
#endif
				if (hasData)
				{
					float opacity = rgbDensityOut.w;
					opacity = rmin(1.0f, opacity);
					rgbBuffer += (1.0f - oBuffer) * make_float3(rgbDensityOut);
					normalBuffer += (1.0f - oBuffer) * opacity * normalOut;
					depthBuffer += (1.0f - oBuffer) * opacity * depthOut;
					oBuffer += (1.0f - oBuffer) * opacity;
				}

				voxelPos = dda.p;
				if (breakOut) break;
			}
#ifndef KERNEL_NO_DEBUG
			if (iter >= maxIter) {
				printf("Infinite loop in DDA, entry=(%f,%f,%f), dir=(%f,%f,%f), pixel=(%d,%d)\n",
					entryPos.x, entryPos.y, entryPos.z,
					rayDir.x, rayDir.y, rayDir.z,
					pixelX, pixelY);
			}
#endif
			
			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};



	//==============================
	// Multi-Iso
	//==============================

	/**
	 * \brief This functor splits the ray inside the voxel into segments
	 * between the TF control points.
	 * Within each segment, the opacity and color are linear functions.
	 * \tparam MarmittNumIterations the number of iterations in Marmitt root finding
	 */
	template<int MarmittNumIterations = 3>
	struct MultiIsoVoxelEval
		: RaySplittingAlgorithm<MultiIsoVoxelEval<MarmittNumIterations>,
				MarmittNumIterations, false>
	{
		/**
		 * \brief Evaluates the contribution at the current control point.
		 * Used for isosurface renderings.
		 *
		 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
		 *

		 * \param volumeTexLinear the texture for normal computation
		 * \param volPos the current position in the volume
		 * \param rayDir the direction of the ray
		 * \param sampleDepth the current sample depth
		 * \param data the color+absorption at the current point
		 * \param rgbBufferOut the blended rgb color (premultiplied with alpha)
		 * \param oBufferOut the blended opacity / alpha
		 */
		static __device__ __inline__ void emitPoint(
			const float3& volPos, const float3& rayDir,
			float sampleDepth, const float4& data,
			float3& rgbBufferOut, float3& normalBufferOut,
			float& depthBufferOut, float& oBufferOut,
			PerPixelInstrumentation* instrumentation)
		{
			//opacityScaling is in percent
			float opacity = min(1.0f, data.w);
			float3 rgb = make_float3(data);

			//compute normal
			float3 normal = make_float3(0, 0, 0);
			if (cDeviceSettings.useShading)
			{
				KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
				const float3& volPos2 = volPos / cDeviceSettings.voxelSize + make_float3(0.5f);
				normal = DvrUtils::computeNormal<TRILINEAR>(volPos2, cDeviceSettings.normalStepSize);
				rgb = DvrUtils::phongShading(rgb, normal, rayDir);
				//printf("volPos=(%.4f, %.4f, %.4f) -> normal=(%.4f, %.4f, %.4f)\n",
				//	volPos2.x, volPos2.y, volPos2.z, normal.x, normal.y, normal.z);
			}

			//blend it
			rgbBufferOut += (1.0f - oBufferOut) * opacity * rgb;
			normalBufferOut += (1.0f - oBufferOut) * opacity * normal;
			depthBufferOut += (1.0f - oBufferOut) * opacity * sampleDepth;
			oBufferOut += (1.0f - oBufferOut) * opacity;
		}
	};

	template<VolumeFilterMode FilterMode>
	struct MultiIsoFixedStepSize
	{
		static __device__ __inline__ void emitPoint(
			const float3& volPos, const float3& rayDir,
			float sampleDepth, const float4& data,
			float3& rgbBufferOut, float3& normalBufferOut,
			float& depthBufferOut, float& oBufferOut,
			PerPixelInstrumentation* instrumentation)
		{
			//opacityScaling is in percent
			float opacity = min(1.0f, data.w);
			float3 rgb = make_float3(data);

			//compute normal
			float3 normal = make_float3(0, 0, 0);
			if (cDeviceSettings.useShading)
			{
				KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
				normal = DvrUtils::computeNormal<TRILINEAR>(volPos, cDeviceSettings.normalStepSize);
				rgb = DvrUtils::phongShading(rgb, normal, rayDir);
				//printf("volPos=(%.4f, %.4f, %.4f) -> normal=(%.4f, %.4f, %.4f)\n",
				//	volPos2.x, volPos2.y, volPos2.z, normal.x, normal.y, normal.z);
			}

			//blend it
			rgbBufferOut += (1.0f - oBufferOut) * opacity * rgb;
			normalBufferOut += (1.0f - oBufferOut) * opacity * normal;
			depthBufferOut += (1.0f - oBufferOut) * opacity * sampleDepth;
			oBufferOut += (1.0f - oBufferOut) * opacity;
		}
		
		/**
		 * The raytracing kernel for DVR (Direct Volume Rendering).
		 *
		 * Input:
		 *  - settings: additional render settings
		 *  - volumeTex: the 3D volume texture
		 *  - tfTexture: transfer function texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Returns RGBA value accumulated through the ray direction.
		 */
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			float3 rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			float oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;

			float3 volPos = (rayStart + fmax(0.0f, tmin) * rayDir - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
			float lastVal = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
			KERNEL_INSTRUMENTATION_INC(densityFetches);
			int lastIndex = cDeviceSettings.tfPoints.searchInterval(lastVal);
			KERNEL_INSTRUMENTATION_INC(tfFetches);
			
			for (real sampleDepth = fmax(0.0f, tmin); sampleDepth < tmax && oBuffer < OpacityEarlyOut; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + float(sampleDepth) * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float currentVal = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				KERNEL_INSTRUMENTATION_INC(densityFetches);
				
				//search for all isosurface crossings
				if (currentVal > lastVal)
				{
					while (cDeviceSettings.tfPoints.positions[lastIndex+1] < currentVal)
					{
						++lastIndex;
						emitPoint(volPos, rayDir, float(sampleDepth), cDeviceSettings.tfPoints.valuesIso[lastIndex], 
							rgbBuffer, normalBuffer, depthBuffer, oBuffer, instrumentation);
						//numerical noise: if the step size is too small, the it quickly jumps between front and back of the isosurface.
						//Hence, after finding an intersection, add a larger step
						sampleDepth += 10 * cDeviceSettings.stepsize;
					}
				} else
				{
					while (cDeviceSettings.tfPoints.positions[lastIndex] >= currentVal)
					{
						emitPoint(volPos, rayDir, float(sampleDepth), cDeviceSettings.tfPoints.valuesIso[lastIndex],
							rgbBuffer, normalBuffer, depthBuffer, oBuffer, instrumentation);
						--lastIndex;
						sampleDepth += 10 * cDeviceSettings.stepsize;
					}
				}
				lastVal = currentVal;
			}

			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};


	//==============================
	// Hybrid: DVR + Multi-Iso
	//==============================

	/**
	 * \brief This functor splits the ray inside the voxel into segments
	 * between the TF control points.
	 * Within each segment, the opacity and color are linear functions.
	 * \tparam MarmittNumIterations the number of iterations in Marmitt root finding
	 */
	template<
		typename IntervalEvaluator,
		int MarmittNumIterations = 3>
	struct HybridVoxelEval
		: RaySplittingAlgorithm<HybridVoxelEval<IntervalEvaluator, MarmittNumIterations>,
		MarmittNumIterations, false>
	{
		/**
		 * \brief Computes the interval
		 *  \f$ \int_{tEntry}^{tExit} L(x) exp(O(x)) dx $\f
		 * where L is the emission and O the absorption.
		 * The mapping from x to density is given by the cubic polynomial 'poly'.
		 * The transfer function is linear in the density, given by
		 *    dataEntry = (L(tf(tEntry)), O(tf(tEntry)))
		 * and
		 *    dataExit = (L(tf(tExit)), O(tf(tExit))) .
		 *
		 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
		 * \param poly
		 * \param tEntry
		 * \param tExit
		 * \param dataEntry
		 * \param dataExit
		 * \param rgbBufferOut the blended rgb color (premultiplied with alpha)
		 * \param oBufferOut the blended opacity / alpha
		 */
		static __host__ __device__ __inline__ void emitInterval(
			const float vals[8], const float4& poly, float tEntry, float tExit,
			const float3& volPos, const float3& rayDir,
			float densityEntry, float densityExit,
			const float4& dataEntry, const float4& dataExit,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (tExit - tEntry < 1e-6) return; //too small, probably divide-by-zero in lerp
			if (dataEntry.w < 1e-6 && dataExit.w < 1e-6) return; //early out

			float3 rgb = { 0,0,0 };
			float opacity = 0;

			auto density = kernel::float4ToPoly<float>(poly);
			auto dataPoly = density.lerp(
				densityEntry, dataEntry,
				densityExit, dataExit);

			//call interval evaluator
			IntervalEvaluator::call(
				vals, dataPoly, tEntry, tExit, volPos, rayDir,
				rgb, opacity, debug, instrumentation);

#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf(" emit density={%.4f, %.4f, %.4f, %.4f}, opacity={%.4f, %.4f, %.4f, %.4f}\n  -> r=%.4f, g=%.4f, b=%.4f, a=%.4f\n",
					density[0], density[1], density[2], density[3],
					dataPoly[0].w, dataPoly[1].w, dataPoly[2].w, dataPoly[3].w,
					rgb.x, rgb.y, rgb.z, opacity);
			}
#endif

			//blend it (this stays the same)
			rgbBufferOut += (1.0f - oBufferOut) * rgb;
			oBufferOut += (1.0f - oBufferOut) * opacity;
		}
		
		/**
		 * \brief Evaluates the contribution at the current control point.
		 * Used for isosurface renderings.
		 *
		 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
		 *

		 * \param volPos the current position in the volume
		 * \param rayDir the direction of the ray
		 * \param sampleDepth the current sample depth
		 * \param data the color+absorption at the current point
		 * \param rgbBufferOut the blended rgb color (premultiplied with alpha)
		 * \param oBufferOut the blended opacity / alpha
		 */
		static __device__ __inline__ void emitPoint(
			const float3& volPos, const float3& rayDir,
			float sampleDepth, const float4& data,
			float3& rgbBufferOut, float3& normalBufferOut,
			float& depthBufferOut, float& oBufferOut,
			PerPixelInstrumentation* instrumentation)
		{
			//opacityScaling is in percent
			float opacity = min(1.0f, data.w);
			float3 rgb = make_float3(data);

			//compute normal
			float3 normal = make_float3(0, 0, 0);
			if (cDeviceSettings.useShading)
			{
				KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
				const float3& volPos2 = volPos / cDeviceSettings.voxelSize + make_float3(0.5f);
				normal = DvrUtils::computeNormal<TRILINEAR>(volPos2, cDeviceSettings.normalStepSize);
				rgb = DvrUtils::phongShading(rgb, normal, rayDir);
				//printf("volPos=(%.4f, %.4f, %.4f) -> normal=(%.4f, %.4f, %.4f)\n",
				//	volPos2.x, volPos2.y, volPos2.z, normal.x, normal.y, normal.z);
			}

			//blend it
			rgbBufferOut += (1.0f - oBufferOut) * opacity * rgb;
			normalBufferOut += (1.0f - oBufferOut) * opacity * normal;
			depthBufferOut += (1.0f - oBufferOut) * opacity * sampleDepth;
			oBufferOut += (1.0f - oBufferOut) * opacity;
		}
	};
	

	//==============================
	// Scale-Invariant DVR
	//==============================

	template<VolumeFilterMode FilterMode>
	struct ScaleInvariantDvrFixedStepSize
	{
		/**
		 * The raytracing kernel for DVR (Direct Volume Rendering).
		 *
		 * Input:
		 *  - settings: additional render settings
		 *  - volumeTex: the 3D volume texture
		 *  - tfTexture: transfer function texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Returns RGBA value accumulated through the ray direction.
		 */
		static __device__ __inline__ RaytraceDvrOutput call(
			int pixelX, int pixelY, 
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			bool debug, PerPixelInstrumentation* instrumentation)
		{
			float3 rgbBuffer = make_float3(0.0f, 0.0f, 0.0f);
			float oBuffer = 0.0f;
			auto normalBuffer = make_float3(0.0f, 0.0f, 0.0f);
			auto depthBuffer = 0.0f;

			float3 volPos = (rayStart + fmax(0.0f, tmin) * rayDir - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
			float lastValue = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
			
			for (real sampleDepth = fmax(0.0f, tmin); sampleDepth < tmax && oBuffer < OpacityEarlyOut; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + float(sampleDepth) * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				KERNEL_INSTRUMENTATION_INC(densityFetches);

				float3 rgb_f;
				float opacity_f;
				KERNEL_INSTRUMENTATION_INC(tfFetches);
				if (!DvrUtils::evalTf(nval, rgb_f, opacity_f))
					continue;

				//this is the core difference:
				// instead of scaling by cDeviceSettings.stepsize
				// scale by abs(nval - lastValue), the approximation of the data
				// values that were traversed
				float alpha = 1 - exp(-opacity_f * abs(nval - lastValue));

				float3 normal = make_float3(0, 0, 0);
				if (cDeviceSettings.useShading)
				{
					// silhouette enhancement
					normal = DvrUtils::computeNormal<FilterMode, false>(volPos, cDeviceSettings.normalStepSize);
					alpha *= length(normal) * 50;

					rgb_f = DvrUtils::phongShading(
						rgb_f, safeNormalize(normal), rayDir);
				}
				
				float3 rgb = rgb_f * alpha;
				lastValue = nval;
				
				rgbBuffer += (1.0f - oBuffer) * rgb;
				depthBuffer += (1.0f - oBuffer) * alpha * float(sampleDepth);
				oBuffer += (1.0f - oBuffer) * alpha;
			}

			return { rgbBuffer, oBuffer, normalBuffer, depthBuffer };
		}
	};

	/**
	 * \brief Implementation of emitInterval() for scale invariant DVR
	 * Within each segment, the opacity and color are linear functions.
	 * \tparam MarmittNumIterations the number of iterations in Marmitt root finding
	 * \tparam ExtendedEarlyOut enables an extra early-out test if the whole
	 *     voxel is in the same zero interval
	 */
	template<
			int MarmittNumIterations = 3,
			bool ExtendedEarlyOut = false>
	struct DvrVoxelEvalScaleInvariant
		: RaySplittingAlgorithm<
			DvrVoxelEvalScaleInvariant<MarmittNumIterations, ExtendedEarlyOut>,
			MarmittNumIterations, ExtendedEarlyOut>
	{
		static __host__ __device__ __inline__ void emitSubinterval(
			float densityEntry, float densityExit, const float4& dataEntry, const float4& dataExit, 
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			float s1, s2, t1, t2;
			float3 c1, c2;
			if (densityEntry<densityExit)
			{
				s1 = densityEntry; s2 = densityExit;
				t1 = dataEntry.w; t2 = dataExit.w;
				c1 = make_float3(dataEntry); c2 = make_float3(dataExit);
			}
			else
			{
				s1 = densityExit; s2 = densityEntry;
				t1 = dataExit.w; t2 = dataEntry.w;
				c1 = make_float3(dataExit); c2 = make_float3(dataEntry);
			}
			if (s1 < 1e-6 || s2 < 1e-6) return;			//fix for errors when numerical inprecisions lead to a wrong selection in the root comparisons
			if (s1 > 1 - 1e-6 || s2 > 1 - 1e-6) return;	//for normal dvr, this is not an issue as this is hidden in the integral
														//But for the integral over the density (not time), combined with the case-3-check, this is crucial
			
			t1 *= cDeviceSettings.opacityScaling; t2 *= cDeviceSettings.opacityScaling;
			
			const Polynomial<1, float> absorption{t1 - s1 * (t2 - t1) / (s2 - s1), (t2 - t1) / (s2 - s1) };
			const Polynomial<2, float> absorptionInt = absorption.integrate();
			const Polynomial<1, float3> color{ c1 - s1 * (c2 - c1) / (s2 - s1), (c2 - c1) / (s2 - s1) };
			const float absorptionInt1 = absorptionInt(s1);
			const float absorptionInt2 = absorptionInt(s2);

			const float opacity = 1-expf(-fabsf(absorptionInt2-absorptionInt1));
			const auto fullEmission = [&color, &absorption, &absorptionInt, &absorptionInt1](float s)
			{
				return color(s) * absorption(s) * expf(-fabsf(absorptionInt(s) - absorptionInt1));
			};
			float allowedError = cDeviceSettings.stepsize * maxCoeff(make_float3(cDeviceSettings.volumeResolution));
			int N = 0;
			float hMax = (s2 - s1);
			float hMin = (s2 - s1) * 1e-2;
			//const float3 rgb = fabs(Quadrature::adaptiveSimpson<decltype(fullEmission), float, float3>(
			//	fullEmission, s1, s2, hMax, allowedError, N, hMin));
			const float3 rgb = fabs(Quadrature::simpson<decltype(fullEmission), float, float3>(
				fullEmission, s1, s2, 10)); N = 10;
			KERNEL_INSTRUMENTATION_INC(intervalEval);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, N);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, N);

#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf(" emit s1=%.4f, s2=%.4f, t1=%.4f, t2=%.2f, r1=%.4f, r2=%.4f\n  -> r=%.4f, g=%.4f, b=%.4f, a=%.4f\n",
					s1, s2, t1, t2, c1.x, c2.y,
					rgb.x, rgb.y, rgb.z, opacity);
			}
#endif

			//blend it (this stays the same)
			rgbBufferOut += (1.0f - oBufferOut) * rgb;
			oBufferOut += (1.0f - oBufferOut) * opacity;
		}

		/**
		 * \brief Computes the scale-invariant volume rendering integral.
		 * The integral is evaluate over the change in density.
		 *
		 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
		 * \param poly
		 * \param tEntry
		 * \param tExit
		 * \param dataEntry
		 * \param dataExit
		 * \param rgbBufferOut the blended rgb color (premultiplied with alpha)
		 * \param oBufferOut the blended opacity / alpha
		 */
		static __host__ __device__ __inline__ void emitInterval(
			const float vals[8], const float4& poly, float tEntry, float tExit,
			const float3& volPos, const float3& rayDir,
			float densityEntry, float densityExit,
			const float4& dataEntry, const float4& dataExit,
			float3& rgbBufferOut, float& oBufferOut,
			const bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (tExit - tEntry < 1e-6) return; //too small, probably divide-by-zero in lerp
			if (dataEntry.w < 1e-6 && dataExit.w < 1e-6) return; //early out

			float s1 = CubicPolynomial<float>::evalCubic(poly, tEntry);
			float s2 = CubicPolynomial<float>::evalCubic(poly, tExit);
			float4 d1 = lerp(dataEntry, dataExit, (s1 - densityEntry) / (densityExit - densityEntry));
			float4 d2 = lerp(dataEntry, dataExit, (s2 - densityEntry) / (densityExit - densityEntry));
			
			static const constexpr bool CheckCase3 = true;
			if constexpr (CheckCase3) {
				//check for case III (double intersection)
				bool isCase3 = false;
				float splitT = 0;
				const auto A = 3 * poly.x, B = 2 * poly.y, C = poly.z;
				auto discr = B * B - 4 * A * C;
				if (discr > 0)
				{
					discr = sqrt(discr);
					float e0 = (-B - fsignf(B) * discr) / (2 * A);
					float e1 = C / (A * e0);
					if (tEntry < e0 && e0 < tExit)
					{
						splitT = e0;
						isCase3 = true;
					}
					else if (tEntry < e1 && e1 < tExit)
					{
						splitT = e1;
						isCase3 = true;
					}
				}
				if (isCase3)
				{
					//case III
					float densitySplit = CubicPolynomial<float>::evalCubic(poly, splitT);
					float4 dataSplit = lerp(dataEntry, dataExit, (densitySplit - densityEntry) / (densityExit - densityEntry));
					emitSubinterval(s1, densitySplit, d1, dataSplit, rgbBufferOut, oBufferOut, debug, instrumentation);
					emitSubinterval(densitySplit, s2, dataSplit, d2, rgbBufferOut, oBufferOut, debug, instrumentation);
				}
				else
				{
					emitSubinterval(densityEntry, densityExit, dataEntry, dataExit, rgbBufferOut, oBufferOut, debug, instrumentation);
				}
			}
			else
				emitSubinterval(s1, s2, d1, d2, rgbBufferOut, oBufferOut, debug, instrumentation);
		}
	};
	
}
