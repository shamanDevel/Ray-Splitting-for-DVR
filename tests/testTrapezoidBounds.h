#pragma once

#include <vector>
#include <cassert>

#include <renderer_math.cuh>
#include <renderer.h>

namespace kernel
{
	typedef PolyExpPoly<6, struct float3, 4, float> PolyType;
	struct PolyWithBounds
	{
		PolyType polynomial;
		float tMin, tMax;
	};
}

std::vector<kernel::PolyWithBounds> launchTrapezoidKernel(
	int screenResolution, const kernel::RendererDeviceSettings& settings,
	cudaTextureObject_t volume_nearest, cudaTextureObject_t volume_linear,
	kernel::OutputTensor& output);

void testIntegrationBounds();
