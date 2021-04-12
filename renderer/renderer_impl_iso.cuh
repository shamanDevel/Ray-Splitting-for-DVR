#pragma once

#include "helper_math.cuh"
#include "renderer_settings.cuh"
#include "renderer_utils.cuh"
#include "renderer_math.cuh"
#include "renderer_math_iso.cuh"

namespace kernel
{
	//=========================================
	// KERNEL IMPLEMENTATIONS - ISO
	//=========================================
	//
	// IsosurfaceKernel (renderer_kernels_iso.cuh)
	//  |-- IsosurfaceFixedStepSize
	//  |
	//  |-- IsosurfaceDDA
	//      |- IsoVoxelEvalFixedSteps
	//      |
	//      | (analytic - all intersections)
	//      |- IsoVoxelEvalAnalyticHyperbolic
	//      |- IsoVoxelEvalAnalyticSchwarze
	//      |- IsoVoxelEvalAnalyticMarmitt
	//      |
	//      | (numeric - first intersection only)
	//      |- IsoVoxelEvalMean
	//      |- IsoVoxelEvalNeubauer
	//      |- IsoVoxelEvalMarmitt
	

	struct RaytraceIsoOutput
	{
		float3 posWorld;
		float3 normalWorld;
		float ao;
		float distance; //time travelled along the ray / distance from start
	};
	
	template<VolumeFilterMode FilterMode>
	struct IsosurfaceFixedStepSize
	{
		/**
		 * The raytracing kernel for isosurfaces.
		 *
		 * Input:
		 *  - screenPos: integer screen position, needed for AO sampling
		 *  - settings: additional render settings
		 *  - volume_tex: the 3D volume texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Return 'true' if an intersection with the isosurface was found.
		 * Then fill 'out' with the position, normal and AO at that position.
		 * Else, return 'false'.
		 */
		static __device__ __inline__ bool call(
			int2 screenPos,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			RaytraceIsoOutput& out, PerPixelInstrumentation* instrumentation)
		{
			bool found = false;
			float3 pos = make_float3(0, 0, 0);
			float3 normal = make_float3(0, 0, 0);
			const float3 volumeResolutionF = make_float3(cDeviceSettings.volumeResolution);
			float sampleDepth;
			for (sampleDepth = max(0.0f, tmin); sampleDepth < tmax && !found; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = rayStart + sampleDepth * rayDir;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				KERNEL_INSTRUMENTATION_INC(densityFetches);
				if (nval > cDeviceSettings.isovalue)
				{
					found = true;
					//TODO: binary search
					//set position to the previous position for AO (slightly outside)
					pos = cDeviceSettings.eyePos + (sampleDepth - cDeviceSettings.stepsize) * rayDir;
					normal.x = 0.5 * (customTex3D(volPos.x + cDeviceSettings.normalStepSize, volPos.y, volPos.z, integral_constant<int, FilterMode>())
						- customTex3D(volPos.x - cDeviceSettings.normalStepSize, volPos.y, volPos.z, integral_constant<int, FilterMode>()));
					normal.y = 0.5 * (customTex3D(volPos.x, volPos.y + cDeviceSettings.normalStepSize, volPos.z, integral_constant<int, FilterMode>())
						- customTex3D(volPos.x, volPos.y - cDeviceSettings.normalStepSize, volPos.z, integral_constant<int, FilterMode>()));
					normal.z = 0.5 * (customTex3D(volPos.x, volPos.y, volPos.z + cDeviceSettings.normalStepSize, integral_constant<int, FilterMode>())
						- customTex3D(volPos.x, volPos.y, volPos.z - cDeviceSettings.normalStepSize, integral_constant<int, FilterMode>()));
					normal = -normal;
					KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
				}
			}
			if (found)
			{
				normal = safeNormalize(normal);
				out.posWorld = pos;
				out.normalWorld = normal;
				out.distance = sampleDepth;
				out.ao = computeAmbientOcclusion<FilterMode>(
					pos - cDeviceSettings.aoBias * rayDir, normal, screenPos.x, screenPos.y);
			}
			return found;
		}
	};

	//Evaluators of the isosurface within a voxel
	//The voxel is defined by the eight surrounding vertices
	//The isosurface is already subtracted, only the intersection needs to be found
	struct IIsoVoxelEval
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
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug);

	protected:

		static __device__ __inline__ float3 normalCenter(const float vals[8])
		{
			float3 normal;
			normal.x = (vals[1] + vals[3] + vals[5] + vals[7] - vals[0] - vals[2] - vals[4] - vals[6]) / 4;
			normal.y = (vals[2] + vals[3] + vals[6] + vals[7] - vals[0] - vals[1] - vals[4] - vals[5]) / 4;
			normal.z = (vals[4] + vals[5] + vals[6] + vals[7] - vals[0] - vals[1] - vals[2] - vals[3]) / 4;
			normal = -normal;
			return normal;
		}

		static __device__ __inline__ float3 normalInterp(const float vals[8], const float3& p)
		{
			float3 normal;
			normal.x =
				lerp(lerp(vals[1], vals[3], p.y), lerp(vals[5], vals[7], p.y), p.z) -
				lerp(lerp(vals[0], vals[2], p.y), lerp(vals[4], vals[6], p.y), p.z);
			normal.y =
				lerp(lerp(vals[2], vals[3], p.x), lerp(vals[6], vals[7], p.x), p.z) -
				lerp(lerp(vals[0], vals[1], p.x), lerp(vals[4], vals[5], p.x), p.z);
			normal.z =
				lerp(lerp(vals[4], vals[5], p.x), lerp(vals[6], vals[7], p.x), p.y) -
				lerp(lerp(vals[0], vals[1], p.x), lerp(vals[2], vals[3], p.x), p.y);
			normal = -normal;
			return normal;
		}
	};

	/*
	 * Mean value of the voxel
	 */
	struct IsoVoxelEvalMean : IIsoVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			float nval = (vals[0] + vals[1] + vals[2] + vals[3] + vals[4] + vals[5] + vals[6] + vals[7]) / 8;
			if (nval > 0)
			{
				timeOut = 0.5 * tExit;
				normalOut = normalCenter(vals);
				return true;
			}
			return false;
		}
	};

	/*
	 * Small steps along the ray
	 */
	template<bool EarlyOut = true>
	struct IsoVoxelEvalFixedSteps : IIsoVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (EarlyOut) {
				if (vals[0] < 0 && vals[1] < 0 && vals[2] < 0 && vals[3] < 0 && vals[4] < 0 && vals[5] < 0 && vals[6] < 0 && vals[7] < 0)
					return false;
				if (vals[0] > 0 && vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] > 0 && vals[5] > 0 && vals[6] > 0 && vals[7] > 0)
				{
					timeOut = 0;
					normalOut = normalInterp(vals, entry);
					return true;
				};
			}

#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf("Fixed Step: t0=%.3f, t1=%.3f\n", 0.0f, tExit);
			}
#endif

			float stepSize = voxelSize.x / 32; //on average 32 steps per voxel
			for (float time=0; time < tExit; time += stepSize)
			{
				float3 p = entry + (time * dir / voxelSize);
				float nval = lerp3D(vals, p);

				if (nval > 0)
				{
					timeOut = time;
					normalOut = normalInterp(vals, p);
					return true;
				}
			}
			return false;
		}
	};

	/*
	 * Analytic ray-iso intersection with the cubic polynom
	 */
	template<typename Scalar_T, bool EarlyOut = true>
	struct IsoVoxelEvalAnalyticHyperbolic : IIsoVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (EarlyOut) {
				if (vals[0] < 0 && vals[1] < 0 && vals[2] < 0 && vals[3] < 0 && vals[4] < 0 && vals[5] < 0 && vals[6] < 0 && vals[7] < 0)
					return false;
				if (vals[0] > 0 && vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] > 0 && vals[5] > 0 && vals[6] > 0 && vals[7] > 0)
				{
					timeOut = 0;
					normalOut = normalInterp(vals, entry);
					return true;
				};
			}
			auto factors = CubicPolynomial<Scalar_T>::getFactors(vals, entry, dir / voxelSize);
			Scalar_T roots[3] = {0};
			int numRoots = CubicPolynomial<Scalar_T>::rootsHyperbolic(factors, roots);
#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf("Analytic: factors: f(x)=%.6f*x^3 + %.6f*x^2 + %.6f*x + %.6f , tMax=%.5f\n",
					float(factors.x), float(factors.y), float(factors.z), float(factors.w), tExit);
				printf("Analytic: num roots: %d, root positions: %.4f (%.4f), %.4f (%.4f), %.4f (%.4f)\n",
					numRoots,
					float(roots[0]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[0])),
					float(roots[1]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[1])),
					float(roots[2]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[2])));
			}
#endif
			//find minimal root in [0, tExit]
			bool found = false;
			float time = tExit;
			for (int i=0; i<numRoots; ++i)
			{
				if (roots[i]>=0 && roots[i]<time
					)
				{
					time = float(roots[i]);
					found = true;
				}
			}
			if (found)
			{
				timeOut = time;
				normalOut = normalInterp(vals, entry + (time * dir / voxelSize));
				return true;
			}
			return false;
		}
	};

	/*
	 * Analytic ray-iso intersection with the cubic polynom
	 */
	template<typename Scalar_T, bool EarlyOut = true>
	struct IsoVoxelEvalAnalyticSchwarze : IIsoVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (EarlyOut) {
				if (vals[0] < 0 && vals[1] < 0 && vals[2] < 0 && vals[3] < 0 && vals[4] < 0 && vals[5] < 0 && vals[6] < 0 && vals[7] < 0)
					return false;
				if (vals[0] > 0 && vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] > 0 && vals[5] > 0 && vals[6] > 0 && vals[7] > 0)
				{
					timeOut = 0;
					normalOut = normalInterp(vals, entry);
					return true;
				};
			}
			auto factors = CubicPolynomial<Scalar_T>::getFactors(vals, entry, dir / voxelSize);
			Scalar_T roots[3] = { 0 };
			int numRoots = CubicPolynomial<Scalar_T>::rootsSchwarze(factors, roots);
#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf("Analytic: factors: f(x)=%.6f*x^3 + %.6f*x^2 + %.6f*x + %.6f , tMax=%.5f\n",
					float(factors.x), float(factors.y), float(factors.z), float(factors.w), tExit);
				printf("Analytic: num roots: %d, root positions: %.4f (%.4f), %.4f (%.4f), %.4f (%.4f)\n",
					numRoots,
					float(roots[0]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[0])),
					float(roots[1]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[1])),
					float(roots[2]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[2])));
			}
#endif
			//find minimal root in [0, tExit]
			bool found = false;
			float time = tExit;
			for (int i = 0; i < numRoots; ++i)
			{
				if (roots[i] >= 0 && roots[i] < time
					)
				{
					time = float(roots[i]);
					found = true;
				}
			}
			if (found)
			{
				timeOut = time;
				normalOut = normalInterp(vals, entry + (time * dir / voxelSize));
				return true;
			}
			return false;
		}
	};

	/*
	 * Analytic ray-iso intersection with the cubic polynom
	 */
	template<typename Scalar_T, bool EarlyOut = true, int NumIterations=5>
	struct IsoVoxelEvalAnalyticMarmitt : IIsoVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (EarlyOut) {
				if (vals[0] < 0 && vals[1] < 0 && vals[2] < 0 && vals[3] < 0 && vals[4] < 0 && vals[5] < 0 && vals[6] < 0 && vals[7] < 0)
					return false;
				if (vals[0] > 0 && vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] > 0 && vals[5] > 0 && vals[6] > 0 && vals[7] > 0)
				{
					timeOut = 0;
					normalOut = normalInterp(vals, entry);
					return true;
				};
			}
			const float3 dir2 = dir / voxelSize;
			auto factors = CubicPolynomial<Scalar_T>::getFactors(vals, entry, dir2);
			float roots[3] = { 0 };
#if 1
			int numRoots = Marmitt<NumIterations, true>::evalAll(
				vals, factors, 0, entry, dir2, 0, tExit, roots);
#else
			bool hasIntersection = Marmitt<NumIterations, true>::eval(
				vals, factors, 0, entry, dir / voxelSize, 0, tExit, roots[0]);
			int numRoots = hasIntersection ? 1 : 0;
#endif
			//int numRoots = CubicPolynomial<Scalar_T>::rootsMarmitt(
			//	factors, roots, 0, tExit);
#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf("Analytic: factors: f(x)=%.6f*x^3 + %.6f*x^2 + %.6f*x + %.6f , tMax=%.5f\n",
					float(factors.x), float(factors.y), float(factors.z), float(factors.w), tExit);
				printf("Analytic: num roots: %d, root positions: %.4f (%.4f), %.4f (%.4f), %.4f (%.4f)\n",
					numRoots,
					float(roots[0]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[0])),
					float(roots[1]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[1])),
					float(roots[2]), float(CubicPolynomial<Scalar_T>::evalCubic(factors, roots[2])));
			}
#endif
			//find minimal root in [0, tExit]
			//Note that the roots are already sorted and in [0, tExit]
			bool found = numRoots > 0;
			float time = roots[0];
			if (found)
			{
				timeOut = time;
				normalOut = normalInterp(vals, entry + (time * dir2));
				return true;
			}
			return false;
		}
	};

	/**
	 * Neubauer's algorithm of repeated linear interpolation.
	 * Can miss some intersections
	 */
	template<bool EarlyOut = true, int NumIterations = 3>
	struct IsoVoxelEvalNeubauer : IIsoVoxelEval
	{
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (EarlyOut) {
				if (vals[0] < 0 && vals[1] < 0 && vals[2] < 0 && vals[3] < 0 && vals[4] < 0 && vals[5] < 0 && vals[6] < 0 && vals[7] < 0)
					return false;
				if (vals[0] > 0 && vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] > 0 && vals[5] > 0 && vals[6] > 0 && vals[7] > 0)
				{
					timeOut = 0;
					normalOut = normalInterp(vals, entry);
					return true;
				};
			}
			const float3 dir2 = dir / voxelSize;

			float t0 = 0, t1 = tExit;
			float v0 = lerp3D(vals, entry), v1 = lerp3D(vals, entry + t1 * dir2);
			if (isign(v0) == isign(v1)) return false; //no hit
			for (int i=0; i<NumIterations; ++i)
			{
				float t = t0 + (t1 - t0) * v0 / (v0 - v1);
				float v = lerp3D(vals, entry + t * dir2);
				if (isign(v)==isign(v0))
				{
					t0 = t;
					v0 = v;
				}
				else
				{
					t1 = t;
					v1 = v;
				}
			}
			timeOut = t0 + (t1 - t0) * v0 / (v0 - v1);
			normalOut = normalInterp(vals, entry + (timeOut * dir2));
			return true;
		}
	};

	/**
	 * Marmitts's algorithm,
	 * improves upon Neubauer by first finding the interval that contains a single intersection.
	 * Then it finds the root using repeated linear interpolation.
	 */
	template<bool EarlyOut = true, int NumIterations = 3, typename PolynomialScalarType = double, bool StableQuadratic = true>
	struct IsoVoxelEvalMarmitt : IIsoVoxelEval
	{
		static __device__ __inline__ void swap(float& a, float& b)
		{
			float c = a; a = b; b = c;
		}
		static __device__ __inline__ bool call(
			const float vals[8], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			if (EarlyOut) {
				if (vals[0] < 0 && vals[1] < 0 && vals[2] < 0 && vals[3] < 0 && vals[4] < 0 && vals[5] < 0 && vals[6] < 0 && vals[7] < 0)
					return false;
				if (vals[0] > 0 && vals[1] > 0 && vals[2] > 0 && vals[3] > 0 && vals[4] > 0 && vals[5] > 0 && vals[6] > 0 && vals[7] > 0)
				{
					timeOut = 0;
					normalOut = normalInterp(vals, entry);
					return true;
				};
			}
			const float3 dir2 = dir / voxelSize;

#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				float t0 = 0, t1 = tExit;
				float v0 = lerp3D(vals, entry), v1 = lerp3D(vals, entry + t1 * dir2);
				printf("Marmitt: t0=%.3f, t1=%.3f, v0=%.3f, v1=%.3f\n", t0, t1, v0, v1);
			}
#endif

			auto factors = CubicPolynomial<PolynomialScalarType>::getFactors(vals, entry, dir2);
			if (Marmitt<NumIterations, StableQuadratic>::eval(
				vals, factors, 0,
				entry, dir2, 0, tExit,
				timeOut
			))
			{
				auto pOut = entry + (timeOut * dir2);
				auto vOut = lerp3D(vals, pOut);
				normalOut = normalInterp(vals, pOut);
				return true;
			}
			else
				return false;
		}
	};

	template<int Neighbors>
	struct IsosurfaceDDA_VoxelFetch;

	template<>
	struct IsosurfaceDDA_VoxelFetch<8>
	{
		static __device__ __inline__ void fetch(
			float vals[8], int3 voxelPos, float isovalue)
		{
			vals[0] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y, voxelPos.z) - isovalue;
			vals[1] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y, voxelPos.z) - isovalue;
			vals[2] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y + 1, voxelPos.z) - isovalue;
			vals[3] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y + 1, voxelPos.z) - isovalue;
			vals[4] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y, voxelPos.z + 1) - isovalue;
			vals[5] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y, voxelPos.z + 1) - isovalue;
			vals[6] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x, voxelPos.y + 1, voxelPos.z + 1) - isovalue;
			vals[7] = tex3D<float>(cDeviceSettings.volumeTexNearest, voxelPos.x + 1, voxelPos.y + 1, voxelPos.z + 1) - isovalue;
		}
	};

	template<>
	struct IsosurfaceDDA_VoxelFetch<64>
	{
		static __device__ __inline__ void fetch(
			float vals[64], int3 voxelPos, float isovalue)
		{
#pragma unroll
			for (int z=0; z<4; ++z) for (int y=0; y<4; ++y) for (int x=0; x<4; ++x)
			{
				vals[x + 4 * (y + 4 * z)] = tex3D<float>(cDeviceSettings.volumeTexNearest,
					voxelPos.x + x - 1, voxelPos.y + y - 1, voxelPos.z + z - 1)
					- isovalue;
			}
		}
	};
	
	/**
	 * \brief Voxel traversal for isosurface rendering.
	 *	It supports 8 and 64-neighborhood.
	 *
	 * The VoxelEvaluation class must contain a function of the type
	 * <code>static __device__ __inline__ bool call(
			const float vals[Neighbors], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug);
		</code>
	 *	
	 * \tparam VoxelEvaluation the voxel evaluation functor.
	 * \tparam Neighbors the number of neighbors, can be 8 (trilinear) or 64 (tricubic)
	 * \tparam SmoothNormals 
	 */
	template<typename VoxelEvaluation, int Neighbors, bool SmoothNormals = true>
	struct IsosurfaceDDA
	{

		/**
		 * The raytracing kernel for isosurfaces using DDA stepping.
		 * Requires a nearest-neighbor sampled volume
		 *
		 * Input:
		 *  - screenPos: integer screen position, needed for AO sampling
		 *  - settings: additional render settings
		 *  - volume_tex: the 3D volume texture
		 *
		 * Input: rayStart, rayDir, tmin, tmax.
		 * The ray enters the volume at rayStart + tmin*rayDir
		 * and leaves the volume at rayStart + tmax*rayDir.
		 *
		 * Output:
		 * Return 'true' if an intersection with the isosurface was found.
		 * Then fill 'out' with the position, normal and AO at that position.
		 * Else, return 'false'.
		 */
		static __device__ __inline__ bool call(
			int2 screenPos,
			const float3& rayStart, const float3& rayDir, float tmin, float tmax,
			RaytraceIsoOutput& out, PerPixelInstrumentation* instrumentation)
		{
			bool found = false;
			float3 pos = make_float3(0, 0, 0);
			float3 normal = make_float3(0, 0, 0);
			float distance = 0;

			//from fixed-step ray-casting as reference:
			//float3 npos = rayStart + sampleDepth * rayDir;
			//float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);

#ifndef KERNEL_NO_DEBUG
			bool debug = cDeviceSettings.pixelSelected && cDeviceSettings.selectedPixel.x == screenPos.x && cDeviceSettings.selectedPixel.y == screenPos.y;
			int3 voxelToTest = make_int3(3, 9, 13);
#else
			bool debug = false;
#endif

			const float3 entryPos = (rayStart + rayDir * (tmin + 1e-7) - cDeviceSettings.boxMin);
			DDA dda(entryPos, rayDir, cDeviceSettings.voxelSize);
			int3 voxelPos = dda.p;
			
#ifndef KERNEL_NO_DEBUG
			if (debug)
			{
				printf("entry voxel position: (%d, %d, %d), ray dir: (%.4f, %.4f, %.4f)\n",
					voxelPos.x, voxelPos.y, voxelPos.z,
					rayDir.x, rayDir.y, rayDir.z);
			}
#endif
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
			
			//main stepping
			const int maxIter = 1<<11;
			int iter;
			for (iter=0;
#ifndef KERNEL_NO_DEBUG
				iter<maxIter &&
#endif
				voxelPos.x>=0 && voxelPos.y>=0 && voxelPos.z>=0 && 
				voxelPos.x<cDeviceSettings.volumeResolution.x-1 && voxelPos.y < cDeviceSettings.volumeResolution.y-1 && voxelPos.z < cDeviceSettings.volumeResolution.z-1;
				iter++)
			{
				//fetch neighbor voxels
				float vals[Neighbors];
				long long timeStart = clock64();
				IsosurfaceDDA_VoxelFetch<Neighbors>::fetch(vals, voxelPos, cDeviceSettings.isovalue);
				KERNEL_INSTRUMENTATION_TIME_ADD(timeDensityFetch, timeStart);
				KERNEL_INSTRUMENTATION_ADD(densityFetches, Neighbors);
				
				//save current pos and time
				const float3 entry = (entryPos + rayDir * dda.t) / cDeviceSettings.voxelSize - make_float3(voxelPos);
				const float tCurrent = dda.t;

				//test if we are already inside the object in the first iteration
				if (iter==0)
				{
					if (lerp3D(vals, entry)>0)
					{
						found = true;
						pos = entryPos + rayDir * tCurrent;
						distance = tmin + tCurrent;
						break;
					}
				}

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
				float timeOut;
#ifndef KERNEL_NO_DEBUG
				if (debug)
				{
					printf("Visit voxel (%d, %d, %d), entry=(%.4f, %.4f, %.4f), tExit=%.4f\n",
						voxelPos.x, voxelPos.y, voxelPos.z, entry.x, entry.y, entry.z, tExit);
				}
				bool debug2 = debug;//&& all(voxelPos == voxelToTest);
				if (debug2) printf("Analyze voxel evaluation for voxel %d, %d, %d\n", voxelPos.x, voxelPos.y, voxelPos.z);
#else
				bool debug2 = false;
#endif
				KERNEL_INSTRUMENTATION_INC(intervalEval);
				found = VoxelEvaluation::call(vals, entry, rayDir, tExit, cDeviceSettings.voxelSize, 
					timeOut, normal, debug2, instrumentation);
				if (found)
				{
					pos = entryPos + rayDir * (tCurrent + timeOut);
					distance = tmin + tCurrent + timeOut;
#ifndef KERNEL_NO_DEBUG
					if (debug)
					{
						printf("Intersection at voxel %d, %d, %d, t=%.8f\n", voxelPos.x, voxelPos.y, voxelPos.z, tCurrent);
					}
#endif
					break;
				}

				voxelPos = dda.p;
				if (breakOut) break;
			}
#ifndef KERNEL_NO_DEBUG
			if (iter >= maxIter) {
				printf("Infinite loop in DDA, entry=(%f,%f,%f), dir=(%f,%f,%f)\n",
					entryPos.x, entryPos.y, entryPos.z,
					rayDir.x, rayDir.y, rayDir.z);
			}
#endif
			
			if (found)
			{
				if (SmoothNormals)
				{
					float3 volPos = pos / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
					normal.x = 0.5 * (customTex3D(volPos.x + cDeviceSettings.normalStepSize, volPos.y, volPos.z, integral_constant<int, TRILINEAR>())
						- customTex3D(volPos.x - cDeviceSettings.normalStepSize, volPos.y, volPos.z, integral_constant<int, TRILINEAR>()));
					normal.y = 0.5 * (customTex3D(volPos.x, volPos.y + cDeviceSettings.normalStepSize, volPos.z, integral_constant<int, TRILINEAR>())
						- customTex3D(volPos.x, volPos.y - cDeviceSettings.normalStepSize, volPos.z, integral_constant<int, TRILINEAR>()));
					normal.z = 0.5 * (customTex3D(volPos.x, volPos.y, volPos.z + cDeviceSettings.normalStepSize, integral_constant<int, TRILINEAR>())
						- customTex3D(volPos.x, volPos.y, volPos.z - cDeviceSettings.normalStepSize, integral_constant<int, TRILINEAR>()));
					normal = -normal;
					KERNEL_INSTRUMENTATION_ADD(densityFetches, 6);
				}
				normal = safeNormalize(normal);
				out.posWorld = pos + cDeviceSettings.boxMin;
				out.normalWorld = normal;
				out.distance = distance;
				out.ao = computeAmbientOcclusion<kernel::VolumeFilterMode::TRILINEAR>(
					out.posWorld - cDeviceSettings.aoBias * rayDir, normal, screenPos.x, screenPos.y);
			}
			return found;
		}
	};
	template<typename VoxelEvaluation, bool SmoothNormals = true>
	using IsosurfaceDDA8 = IsosurfaceDDA<VoxelEvaluation, 8, SmoothNormals>;

	template<typename VoxelEvaluation, bool SmoothNormals = true>
	using IsosurfaceDDA64 = IsosurfaceDDA<VoxelEvaluation, 64, SmoothNormals>;


	//################################
	// TRICUBIC INTERPOLATION
	//################################
	
	/*
	 * Small steps along the ray (with polygon)
	 */
	template<TricubicFactorAlgorithm FactorAlgorithm>
	struct IsoTricubicVoxelEvalFixedSteps
	{
		static __device__ __inline__ bool call(
			const float vals[64], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			//build polynomial
			long long start = clock64();
			const auto nonicPoly = TricubicInterpolation<float>::getFactors(
				vals, entry, dir / voxelSize,
				integral_constant<TricubicFactorAlgorithm, FactorAlgorithm>());
			KERNEL_INSTRUMENTATION_TIME_ADD(timePolynomialCreation, start);

			////test
			//float3 exit = entry + tExit * (dir / voxelSize);
			//printf("entry=(%.4f, %.4f, %.4f), exit=(%.4f, %.4f, %.4f)\n",
			//	entry.x, entry.y, entry.z, exit.x, exit.y, exit.z);

			start = clock64();
			float stepSize = cDeviceSettings.stepsize; // voxelSize.x / 32; //on average 32 steps per voxel
			bool found = false;
			int numSteps = 0;
			for (float time = 0; time < tExit; time += stepSize)
				//for (float time = 0; ; time += stepSize)
			{
				float nval = nonicPoly(time);
				numSteps++;

				if (nval > 0)
				{
					timeOut = time;
					//normalOut = normalInterp(vals, p);
					found = true;
					break;
				}
			}
			KERNEL_INSTRUMENTATION_TIME_ADD(timePolynomialSolution, start);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, numSteps);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, numSteps);
			return found;
		}
	};

	/*
	 * Small steps along the ray (no nonic polynomial)
	 */
	struct IsoTricubicVoxelEvalFixedStepsNoPoly
	{
		static __device__ __inline__ bool call(
			const float vals[64], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			//build polynomial
			long long start = clock64();
			const auto interpolation = TricubicInterpolation<float>(
				entry, dir / voxelSize);
			KERNEL_INSTRUMENTATION_TIME_ADD(timePolynomialCreation, start);

			start = clock64();
			float stepSize = cDeviceSettings.stepsize; // voxelSize.x / 32; //on average 32 steps per voxel
			bool found = false;
			int numSteps = 0;
			for (float time = 0; time < tExit; time += stepSize)
				//for (float time = 0; ; time += stepSize)
			{
				float nval = interpolation.call(vals, time);
				numSteps++;

				if (nval > 0)
				{
					timeOut = time;
					//normalOut = normalInterp(vals, p);
					found = true;
					break;
				}
			}
			KERNEL_INSTRUMENTATION_TIME_ADD(timePolynomialSolution, start);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, numSteps);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, numSteps);
			return found;
		}
	};

	/*
	 * Sphere Tracing for the first intersection
	 *
	 * \tparam BoundsFunctor_t the functor to compute Lipschitz bounds.
	 *   Currently supported: \ref SimpleBound and \ref BernsteinBound
	 */
	template<typename BoundsFunctor_t, TricubicFactorAlgorithm FactorAlgorithm>
	struct IsoTricubicVoxelEvalSphereTracing
	{
		static __device__ __inline__ bool call(
			const float vals[64], const float3& entry, const float3& dir, float tExit, const float3& voxelSize,
			float& timeOut, float3& normalOut, bool debug, PerPixelInstrumentation* instrumentation)
		{
			//build polynomial
			long long start = clock64();
			const auto nonicPoly = TricubicInterpolation<float>::getFactors(
				vals, entry, dir / voxelSize, 
				integral_constant<TricubicFactorAlgorithm, FactorAlgorithm>());
			KERNEL_INSTRUMENTATION_TIME_ADD(timePolynomialCreation, start);

			start = clock64();
			float epsilon = 1e-6f;
			auto ti = sphereTrace(
				nonicPoly,
				tExit,
				BoundsFunctor_t(),
				epsilon,
				debug);
			float t = ti.first;
			KERNEL_INSTRUMENTATION_TIME_ADD(timePolynomialSolution, start);
			KERNEL_INSTRUMENTATION_ADD(intervalStep, ti.second);
			KERNEL_INSTRUMENTATION_MAX(intervalMaxStep, ti.second);
			timeOut = t;
			return t >= 0;
		}
	};
}