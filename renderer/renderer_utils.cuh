#pragma once

#include "helper_math.cuh"
#include "renderer_settings.cuh"
#include "renderer_commons.cuh"

__constant__ float4 cAOHemisphere[MAX_AMBIENT_OCCLUSION_SAMPLES];
__constant__ float4 cAORandomRotations[AMBIENT_OCCLUSION_RANDOM_ROTATIONS * AMBIENT_OCCLUSION_RANDOM_ROTATIONS];
__constant__ kernel::RendererDeviceSettings cDeviceSettings;

namespace kernel
{
	
	//=========================================
	// RENDERER UTILITIES
	//=========================================

	inline __device__ float4 matmul(const float4 mat[4], float4 v)
	{
		return make_float4(
			dot(mat[0], v),
			dot(mat[1], v),
			dot(mat[2], v),
			dot(mat[3], v)
		);
	}

	__device__ __forceinline__ float customTex3D(float x, float y, float z,
		integral_constant<int, NEAREST>)
	{
		return tex3D<float>(cDeviceSettings.volumeTexNearest, x, y, z);
	}
	
	__device__ __forceinline__ float customTex3D(float x, float y, float z,
		integral_constant<int, TRILINEAR>)
	{
		return tex3D<float>(cDeviceSettings.volumeTexLinear, x, y, z);
	}

	//Source: https://github.com/DannyRuijters/CubicInterpolationCUDA
	// Inline calculation of the bspline convolution weights, without conditional statements
	template<class T> inline __device__ void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
	{
		const T one_frac = 1.0f - fraction;
		const T squared = fraction * fraction;
		const T one_sqd = one_frac * one_frac;

		w0 = 1.0f / 6.0f * one_sqd * one_frac;
		w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
		w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
		w3 = 1.0f / 6.0f * squared * fraction;
	}
	//Source: https://github.com/DannyRuijters/CubicInterpolationCUDA
	//TODO: change to texture object API to support char, short and float textures
	__device__ __forceinline__ float customTex3D(float x, float y, float z,
		integral_constant<int, TRICUBIC>)
	{
		const float3 coord = make_float3(x, y, z);
		// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
		const float3 coord_grid = coord - 0.5f;
		const float3 index = floorf(coord_grid);
		const float3 fraction = coord_grid - index;
		float3 w0, w1, w2, w3;
		bspline_weights(fraction, w0, w1, w2, w3);

		const float3 g0 = w0 + w1;
		const float3 g1 = w2 + w3;
		const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
		const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

		// fetch the eight linear interpolations
		// weighting and fetching is interleaved for performance and stability reasons
		typedef float floatN; //return type
		floatN tex000 = tex3D<float>(cDeviceSettings.volumeTexLinear, h0.x, h0.y, h0.z);
		floatN tex100 = tex3D<float>(cDeviceSettings.volumeTexLinear, h1.x, h0.y, h0.z);
		tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
		floatN tex010 = tex3D<float>(cDeviceSettings.volumeTexLinear, h0.x, h1.y, h0.z);
		floatN tex110 = tex3D<float>(cDeviceSettings.volumeTexLinear, h1.x, h1.y, h0.z);
		tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
		tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
		floatN tex001 = tex3D<float>(cDeviceSettings.volumeTexLinear, h0.x, h0.y, h1.z);
		floatN tex101 = tex3D<float>(cDeviceSettings.volumeTexLinear, h1.x, h0.y, h1.z);
		tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
		floatN tex011 = tex3D<float>(cDeviceSettings.volumeTexLinear, h0.x, h1.y, h1.z);
		floatN tex111 = tex3D<float>(cDeviceSettings.volumeTexLinear, h1.x, h1.y, h1.z);
		tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
		tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

		return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
	}

	__host__ __device__ __forceinline__ void intersectionRayAABB(
		const float3& rayStart, const float3& rayDir,
		const float3& boxMin, const float3& boxSize,
		float& tmin, float& tmax)
	{
		float3 invRayDir = 1.0f / rayDir;
		float t1 = (boxMin.x - rayStart.x) * invRayDir.x;
		float t2 = (boxMin.x + boxSize.x - rayStart.x) * invRayDir.x;
		float t3 = (boxMin.y - rayStart.y) * invRayDir.y;
		float t4 = (boxMin.y + boxSize.y - rayStart.y) * invRayDir.y;
		float t5 = (boxMin.z - rayStart.z) * invRayDir.z;
		float t6 = (boxMin.z + boxSize.z - rayStart.z) * invRayDir.z;
		tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
		tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
	}
	__host__ __device__ __forceinline__ void intersectionRayAABB(
		const double3& rayStart, const double3& rayDir,
		const double3& boxMin, const double3& boxSize,
		double& tmin, double& tmax)
	{
		double3 invRayDir = 1.0f / rayDir;
		double t1 = (boxMin.x - rayStart.x) * invRayDir.x;
		double t2 = (boxMin.x + boxSize.x - rayStart.x) * invRayDir.x;
		double t3 = (boxMin.y - rayStart.y) * invRayDir.y;
		double t4 = (boxMin.y + boxSize.y - rayStart.y) * invRayDir.y;
		double t5 = (boxMin.z - rayStart.z) * invRayDir.z;
		double t6 = (boxMin.z + boxSize.z - rayStart.z) * invRayDir.z;
		tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
		tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
	}



	template<VolumeFilterMode FilterMode>
	__device__ float computeAmbientOcclusion(
		float3 pos, float3 normal,
		int x, int y)
	{
		if (cDeviceSettings.aoSamples == 0) return 1;
		float ao = 0.0;
		//get random rotation vector
		int x2 = x % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
		int y2 = y % AMBIENT_OCCLUSION_RANDOM_ROTATIONS;
		float3 noise = make_float3(cAORandomRotations[x2 + AMBIENT_OCCLUSION_RANDOM_ROTATIONS * y2]);
		//compute transformation
		float3 tangent = normalize(noise - normal * dot(noise, normal));
		float3 bitangent = cross(normal, tangent);
		//sample
		const float bias = cDeviceSettings.isovalue;//customTex3D(volume_tex, pos.x, pos.y, pos.z, std::integral_constant<int, FilterMode>());
		for (int i = 0; i < cDeviceSettings.aoSamples; ++i)
		{
			//get hemisphere sample
			float3 sampleT = normalize(make_float3(cAOHemisphere[i]));
			//transform to world space
			float3 sampleW = make_float3(
				dot(make_float3(tangent.x, bitangent.x, normal.x), sampleT),
				dot(make_float3(tangent.y, bitangent.y, normal.y), sampleT),
				dot(make_float3(tangent.z, bitangent.z, normal.z), sampleT)
			);
			//shoot ray
			float tmin, tmax;
			intersectionRayAABB(pos, sampleW, cDeviceSettings.boxMin, cDeviceSettings.boxSize, tmin, tmax);
			//assert(tmax > 0 && tmin < tmax);
			tmax = min(tmax, cDeviceSettings.aoRadius);
			float value = 1.0;
			const float3 volumeResolutionF = make_float3(cDeviceSettings.volumeResolution);
			for (float sampleDepth = cDeviceSettings.stepsize; sampleDepth <= tmax; sampleDepth += cDeviceSettings.stepsize)
			{
				float3 npos = pos + sampleDepth * sampleW;
				float3 volPos = (npos - cDeviceSettings.boxMin) / cDeviceSettings.voxelSize + make_float3(0.5, 0.5, 0.5);
				float nval = customTex3D(volPos.x, volPos.y, volPos.z, integral_constant<int, FilterMode>());
				if (nval > bias)
				{
					value = .0f;//smoothstep(1, 0, cDeviceSettings.aoRadius / sampleDepth);
					break;
				}
			}
			ao += value;
		}
		return ao / cDeviceSettings.aoSamples;
	}

#ifndef DDA_USE_DOUBLE
#define DDA_USE_DOUBLE 0
#endif

// algorithm by Jozsa (2014) "Analytic Isosurface Rendering and Maximum Intensity Projection on the GPU"
#ifndef DDA_USE_JOZSA
#define DDA_USE_JOZSA 1
#endif
	
	struct DDA
	{
	public: //public for debugging only
		int3 pStep;
#if DDA_USE_DOUBLE==1
		double3 tDel, tSide;
#else
		float3 tDel, tSide;
		float tOffset = 0;
#endif
	public:
		/**
		 * The time since entering the grid
		 */
		float t;
		/**
		 * The current voxel position
		 */
		int3 p;

#if DDA_USE_JOZSA
		int3 iOffset = make_int3(0, 0, 0);
#endif

		/**
		 * \brief Prepares the DDA traversal.
		 *   the variable 'p' will contain the voxel position of the entry
		 * \param entry the entry position into the volume
		 * \param dir the ray direction
		 * \param voxelSize the size of the voxels
		 */
		__host__ __device__ DDA(const float3& entry, const float3 dir, const float3& voxelSize)
		{
#if DDA_USE_DOUBLE==1
			double3 pStepF = fsign(make_double3(dir));
			double3 pF = make_double3(entry) / make_double3(voxelSize);
			tDel = fabs(make_double3(voxelSize) / make_double3(dir));
			tSide = ((floor(pF) - pF + 0.5)*pStepF + 0.5) * tDel;
			pStep = make_int3(pStepF);
			p = make_int3(floor(pF));
#else
			float3 pStepF = fsignf(dir);
			float3 pF = entry / voxelSize;
			tDel = fabs(voxelSize / dir);
			tSide = ((floorf(pF) - pF + 0.5) * pStepF + 0.5) * tDel;
			pStep = make_int3(pStepF);
			p = make_int3(floorf(pF));
#endif
			t = 0;
		}

		/**
		 * Steps into the next voxel along the ray. The variables 'p' and 't' are updated
		 */
		__host__ __device__ void step()
		{
#if DDA_USE_JOZSA

			float3 localTSide = tSide + make_float3(iOffset) * tDel;
			int3 mask;
			mask.x = (localTSide.x < localTSide.y) & (localTSide.x <= localTSide.z);
			mask.y = (localTSide.y < localTSide.z) & (localTSide.y <= localTSide.x);
			mask.z = (localTSide.z < localTSide.x) & (localTSide.z <= localTSide.y);
#ifndef KERNEL_NO_DEBUG
			if (mask.x == 0 && mask.y == 0 && mask.z == 0)
			{
				printf("DDA ERROR: no stepping performed! tSide=(%f,%f,%f), tDel=(%.3f,%.3f,%.3f), pStep=(%d,%d,%d)\n",
					tSide.x, tSide.y, tSide.z, tDel.x, tDel.y, tDel.z, pStep.x, pStep.y, pStep.z);
			}
#endif
			t = (mask.x ? localTSide.x : (mask.y ? localTSide.y : localTSide.z));
			iOffset += mask;
			p += mask * pStep;
			
#else
			int3 mask;
			mask.x = (tSide.x < tSide.y) & (tSide.x <= tSide.z);
			mask.y = (tSide.y < tSide.z) & (tSide.y <= tSide.x);
			mask.z = (tSide.z < tSide.x) & (tSide.z <= tSide.y);
#ifndef KERNEL_NO_DEBUG
			if (mask.x==0 && mask.y==0 && mask.z==0)
			{
				printf("DDA ERROR: no stepping performed! tSide=(%f,%f,%f), tDel=(%.3f,%.3f,%.3f), pStep=(%d,%d,%d)\n",
					tSide.x, tSide.y, tSide.z, tDel.x, tDel.y, tDel.z, pStep.x, pStep.y, pStep.z);
			}
#endif
			
#if DDA_USE_DOUBLE==1
			t = mask.x ? tSide.x : (mask.y ? tSide.y : tSide.z);
			tSide += make_double3(mask) * tDel;
#else
#if 1
			//apply offset, this reduces the numerical errors
			t = tOffset + (mask.x ? tSide.x : (mask.y ? tSide.y : tSide.z));
			tSide += make_float3(mask) * tDel;
			float m = minimum(tSide.x, tSide.y, tSide.z);
			tOffset += m;
			tSide -= m;
#else
			//original method
			t = (mask.x ? tSide.x : (mask.y ? tSide.y : tSide.z));
			tSide += make_float3(mask) * tDel;
#endif
#endif
			p += mask * pStep;
#endif
		}
	};


	
}