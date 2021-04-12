#include <catch.hpp>
#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

//define to avoid errors when compiling without CUDA
#ifndef __NVCC
template<typename T> T tex1D(cudaTextureObject_t, float) { throw std::exception("not implemented"); }
template<typename T> T tex3D(cudaTextureObject_t, float, float, float) { throw std::exception("not implemented"); }
#endif

#include <renderer_impl_dvr.cuh>

TEST_CASE("DvrVoxelEvalInterval-Random", "[Dvr][!hide]")
{
	std::default_random_engine rnd(0);
	std::uniform_real_distribution<float> distr1(0, 1);
	std::uniform_int_distribution<int> distr2(0, 1);

	//create settings
	kernel::RendererDeviceSettings settings;
	settings.voxelSize = make_float3(1);
	//a single peak
	float tfPositions[] = { 0.0, 0.4, 0.5, 0.6, 1.0 };
	float4 tfValues[] = {
		make_float4(0,0,0, 0.0),
		make_float4(1,0,0, 0.0),
		make_float4(0,1,0, 1.0),
		make_float4(0,0,1, 0.0),
		make_float4(0,0,0, 0.0)
	};
	int N = sizeof(tfPositions) / sizeof(float);
	settings.tfPoints.numPoints = N;
	for (int j = 0; j < N; ++j) settings.tfPoints.positions[j] = tfPositions[j];
	for (int j = 0; j < N; ++j) settings.tfPoints.values[j] = tfValues[j];
	
	for (int i=0; i<50; ++i)
	{
		std::cout << "i=" << i << std::endl;
		//create random test case
		float vals[8];
		std::cout << "Corner values:";
		for (int j = 0; j < 8; ++j) {
			vals[j] = distr1(rnd);
			std::cout << " " << vals[j] << std::endl;
		}
		std::cout << std::endl;
		
		int edgeIndexIn[3] = { 0,1,2 };
		std::shuffle(edgeIndexIn, edgeIndexIn + 3, rnd);
		float entry[3];
		entry[edgeIndexIn[0]] = distr1(rnd);
		entry[edgeIndexIn[1]] = distr1(rnd);
		entry[edgeIndexIn[2]] = distr2(rnd);
		int edgeIndexOut[3] = { 0,1,2 };
		do {
			std::shuffle(edgeIndexOut, edgeIndexOut + 3, rnd);
		} while (edgeIndexIn[0] == edgeIndexOut[0] && edgeIndexIn[1] == edgeIndexOut[1]);
		float exit[3];
		exit[edgeIndexOut[0]] = distr1(rnd);
		exit[edgeIndexOut[1]] = distr1(rnd);
		exit[edgeIndexOut[2]] = distr2(rnd);
		float3 entryV = make_float3(entry[0], entry[1], entry[2]);
		float3 exitV = make_float3(exit[0], exit[1], exit[2]);

		float3 dir = exitV - entryV;
		float tExit = 1;

		std::cout << "Entry: " << entryV.x << ", " << entryV.y << ", " << entryV.z
			<< "  with value " << kernel::lerp3D(vals, entryV) << std::endl;
		std::cout << "Exit: " << exitV.x << ", " << exitV.y << ", " << exitV.z
			<< "  with value " << kernel::lerp3D(vals, exitV) << std::endl;
		std::cout << "Dir: " << dir.x << ", " << dir.y << ", " << dir.z << std::endl;
		{
			double4 factors = kernel::CubicPolynomial<double>::getFactors(vals, entryV, dir);
			for (int j=0; j<N; ++j)
			{
				double4 f2 = factors;
				f2.w -= settings.tfPoints.positions[j];
				double roots[3];
				int numRoots = kernel::CubicPolynomial<double>::rootsHyperbolic(f2, roots);
				std::cout << "Roots for density " << settings.tfPoints.positions[j] << ":\n";
				for (int k = 0; k < numRoots; ++k)
					std::cout << "  t=" << roots[k] << "  ("
						<< kernel::CubicPolynomial<double>::evalCubic(factors, roots[k]) << ")\n";
				std::cout << std::flush;
			}
		}
		
		float4 rgbDensity = make_float4(0);
		float3 normal = make_float3(0);
		float depth = 0;

		bool result = kernel::DvrVoxelEvalInterval::call(
			vals, entryV, dir, tExit, settings, {}, rgbDensity, normal, depth, false);
		std::cout << "intersection: " << result << std::endl;

		std::cout << std::endl;
	}
}

TEST_CASE("DvrVoxelEvalInterval-Single", "[Dvr][!hide]")
{

	//create settings
	kernel::RendererDeviceSettings settings;
	settings.voxelSize = make_float3(1);
	settings.opacityScaling = 50;
	settings.voxelSize = make_float3(1 / 16.0f);
	//a single peak
	float tfPositions[] = { 0.0, 0.4, 0.5, 0.6, 1.0 };
	float4 tfValues[] = {
		make_float4(0,0,0, 0.0),
		make_float4(1,0,0, 0.0),
		make_float4(0,1,0, 1.0),
		make_float4(0,0,1, 0.0),
		make_float4(0,0,0, 0.0)
	};
	int N = sizeof(tfPositions) / sizeof(float);
	settings.tfPoints.numPoints = N;
	for (int j = 0; j < N; ++j) settings.tfPoints.positions[j] = tfPositions[j];
	for (int j = 0; j < N; ++j) settings.tfPoints.values[j] = tfValues[j];

	float vals[8] = { 0.4599, 0.5213, 0.3386, 0.3876, 0.4796, 0.5436, 0.3545, 0.4049 };
	std::cout << "Corner values:";
	for (int j = 0; j < 8; ++j) {
		std::cout << " " << vals[j] << std::endl;
	}
	std::cout << std::endl;

	float3 entryV{ 1.0000, 0.6255, 0.2073 };
	float3 dir{ -8.8187, -9.2572, 7.8444 };
	float tExit = 0.0676;
	float3 exitV = entryV + dir * tExit;

	std::cout << "Entry: " << entryV.x << ", " << entryV.y << ", " << entryV.z
		<< "  with value " << kernel::lerp3D(vals, entryV) << std::endl;
	std::cout << "Exit: " << exitV.x << ", " << exitV.y << ", " << exitV.z
		<< "  with value " << kernel::lerp3D(vals, exitV) << std::endl;
	std::cout << "Dir: " << dir.x << ", " << dir.y << ", " << dir.z << std::endl;
	{
		double4 factors = kernel::CubicPolynomial<double>::getFactors(vals, entryV, dir/ settings.voxelSize);
		for (int j = 0; j < N; ++j)
		{
			double4 f2 = factors;
			f2.w -= settings.tfPoints.positions[j];
			double roots[3];
			int numRoots = kernel::CubicPolynomial<double>::rootsHyperbolic(f2, roots);
			std::cout << "Roots for density " << settings.tfPoints.positions[j] << ":\n";
			for (int k = 0; k < numRoots; ++k)
				std::cout << "  t=" << roots[k] << "  ("
				<< kernel::CubicPolynomial<double>::evalCubic(factors, roots[k]) << ")\n";
			std::cout << std::flush;
		}
	}

	float4 rgbDensity = make_float4(0);
	float3 normal = make_float3(0);
	float depth = 0;

	bool result = kernel::DvrVoxelEvalInterval::call(
		vals, entryV, dir, tExit, settings, {}, rgbDensity, normal, depth, false);
	std::cout << "intersection: " << result << std::endl;

	std::cout << std::endl;
}