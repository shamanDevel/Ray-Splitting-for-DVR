#pragma once
/**
 * Kernel definitions, included either in renderer_static.cu or renderer_rtc_kernels.cu
 */

#include "helper_math.cuh"
#include "renderer_settings.cuh"
#include "renderer_utils.cuh"
#include "renderer_impl_iso.cuh"
#include "renderer_color.cuh"

//Defined if running in pre-compilation mode using the runtime API.
// It creates the launcher for the kernels then.
//Not defined if running in runtime-compilation mode.
// The kernels are accessed via the driver API then.
#ifndef REGISTER_KERNEL
#define REGISTER_KERNEL(name, ...)
#endif

namespace kernel
{
	//=========================================
	// MAIN ENTRY FUNCTIONS
	//=========================================

	__device__ inline void writeOutputIso(
		OutputTensor& output, int x, int y,
		float mask, float3 normal, float depth, float ao, float2 flow)
	{
		output.coeff(y, x, 0) = mask;
		output.coeff(y, x, 1) = normal.x;
		output.coeff(y, x, 2) = normal.y;
		output.coeff(y, x, 3) = normal.z;
		output.coeff(y, x, 4) = depth;
		output.coeff(y, x, 5) = ao;
		output.coeff(y, x, 6) = flow.x;
		output.coeff(y, x, 7) = flow.y;
	}
	__device__ inline void writeOutputIso(
		OutputTensor& output, int x, int y)
	{
		writeOutputIso(output, x, y, 0, make_float3(0, 0, 0), 0, 1, make_float2(0, 0));
	}

	/**
	 * The rendering kernel with the parallel loop over the pixels and output handling
	 */
	template<typename Functor>
	__global__ void IsosurfaceKernel(dim3 virtual_size,
		OutputTensor output, PerPixelInstrumentation* instrumentationOut)
	{
		for (ptrdiff_t __i = blockIdx.x * blockDim.x + threadIdx.x; __i < virtual_size.x*virtual_size.y; __i += blockDim.x * gridDim.x) {
			const ptrdiff_t y_ = __i / virtual_size.x;
			const ptrdiff_t x_ = __i - y_ * virtual_size.x;

			//prepare instrumentation
#if KERNEL_INSTRUMENTATION==1
			PerPixelInstrumentation localInstrumentation = { 0 };
#else
			PerPixelInstrumentation localInstrumentation;
#endif
			PerPixelInstrumentation* instrumentation = &localInstrumentation;
			
			int x = x_ + cDeviceSettings.viewport.x;
			int y = y_ + cDeviceSettings.viewport.y;
			float posx = ((x + 0.5f) / cDeviceSettings.screenSize.x) * 2 - 1;
			float posy = ((y + 0.5f) / cDeviceSettings.screenSize.y) * 2 - 1;

			//target world position
			float4 screenPos = make_float4(posx, posy, 0.9, 1);
			float4 worldPos = matmul(cDeviceSettings.currentViewMatrixInverse, screenPos);
			worldPos /= worldPos.w;

			//ray direction
			float3 rayDir = normalize(make_float3(worldPos) - cDeviceSettings.eyePos);

			//entry, exit points
			float tmin, tmax;
			intersectionRayAABB(cDeviceSettings.eyePos, rayDir, cDeviceSettings.boxMin, cDeviceSettings.boxSize, tmin, tmax);

			//clip plane
			if (cDeviceSettings.enableClipPlane)
			{
				float3 entryPos = cDeviceSettings.eyePos;
				float nom = dot(make_float3(cDeviceSettings.clipPlane), entryPos);
				float denom = dot(make_float3(cDeviceSettings.clipPlane), rayDir);
				if (fabs(denom) > 1e-5)
				{
					float t = -(nom + cDeviceSettings.clipPlane.w) / denom;
					if (denom > 0)
						tmin = fmaxf(tmin, t);
					else
						tmax = fminf(tmax, t);
				}
			}

			if (tmax < 0 || tmin > tmax)
			{
				writeOutputIso(output, x_, y_);
#if KERNEL_INSTRUMENTATION==1
				if (instrumentationOut) instrumentationOut[__i] = localInstrumentation;
#endif
				continue;
			}

			//perform stepping
			RaytraceIsoOutput out;
			long long start = clock64();
			bool found = Functor::call(
				make_int2(x, y), cDeviceSettings.eyePos, rayDir, tmin, tmax, out, instrumentation);
			KERNEL_INSTRUMENTATION_TIME_ADD0(timeTotal, start);
			if (found)
			{
#ifndef KERNEL_NO_DEBUG
				if (cDeviceSettings.pixelSelected && cDeviceSettings.selectedPixel.x == x && cDeviceSettings.selectedPixel.y == y)
				{
					//debug print
					printf("world position: %.4f, %.4f, %.4f\nnormal: %.4f, %.4f, %.4f\n",
						out.posWorld.x, out.posWorld.y, out.posWorld.z, out.normalWorld.x, out.normalWorld.y, out.normalWorld.z);
				}
#endif

				float4 screenCurrent = matmul(cDeviceSettings.currentViewMatrix, make_float4(out.posWorld, 1.0));
				screenCurrent /= screenCurrent.w;
				float4 screenNext = matmul(cDeviceSettings.nextViewMatrix, make_float4(out.posWorld, 1.0));
				screenNext /= screenNext.w;
				//evaluate depth and flow
				float mask = 1;
				float depth = out.distance;//screenCurrent.z;
				float2 flow = 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y);
				float3 normalScreen = make_float3(matmul(cDeviceSettings.normalMatrix, make_float4(-out.normalWorld, 0)));
				//write output
				writeOutputIso(output, x_, y_, mask, normalScreen, depth, out.ao, flow);
			}
			else
			{
#ifndef KERNEL_NO_DEBUG
				if (cDeviceSettings.pixelSelected && cDeviceSettings.selectedPixel.x == x && cDeviceSettings.selectedPixel.y == y)
				{
					printf("no intersection with the isosurface found\n");
				}
#endif
				writeOutputIso(output, x_, y_);
			}
#if KERNEL_INSTRUMENTATION==1
			if (instrumentationOut) instrumentationOut[__i] = localInstrumentation;
#endif
		}
	}

	//commented out unneeded cases to reduce compile time

	REGISTER_KERNEL("Iso: Fixed step size - nearest", kernel::IsosurfaceKernel<kernel::IsosurfaceFixedStepSize<kernel::NEAREST>>);
	REGISTER_KERNEL("Iso: Fixed step size - trilinear", kernel::IsosurfaceKernel<kernel::IsosurfaceFixedStepSize<kernel::TRILINEAR>>);
	REGISTER_KERNEL("Iso: Fixed step size - tricubic", kernel::IsosurfaceKernel<kernel::IsosurfaceFixedStepSize<kernel::TRICUBIC>>);
	
	REGISTER_KERNEL("Iso: DDA - fixed step", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalFixedSteps<> >>);
	
	REGISTER_KERNEL("Iso: DDA - [ana] hyperbolic (float)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalAnalyticHyperbolic<float> >>);
	//REGISTER_KERNEL("Iso: DDA - [ana] hyperbolic (double)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA<kernel::IsoVoxelEvalAnalyticHyperbolic<double> >>);
	REGISTER_KERNEL("Iso: DDA - [ana] Schwarze (float)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalAnalyticSchwarze<float> >>);
	//REGISTER_KERNEL("Iso: DDA - [ana] Schwarze (double)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA<kernel::IsoVoxelEvalAnalyticSchwarze<double> >>);
	REGISTER_KERNEL("Iso: DDA - [ana] Marmitt (float)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalAnalyticMarmitt<float, true, 4> >>);
	//REGISTER_KERNEL("Iso: DDA - [ana] Marmitt (double)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA<kernel::IsoVoxelEvalAnalyticMarmitt<double, true, 4> >>);
	
	REGISTER_KERNEL("Iso: DDA - [num] midpoint", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalMean, false>>);
	REGISTER_KERNEL("Iso: DDA - [num] linear", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalNeubauer<true, 0> >>);
	REGISTER_KERNEL("Iso: DDA - [num] Neubauer", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalNeubauer<> >>);
	REGISTER_KERNEL("Iso: DDA - [num] Marmitt (float, unstable)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalMarmitt<true, 4, float, false> >>);
	//REGISTER_KERNEL("Iso: DDA - [num] Marmitt (double, unstable)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalMarmitt<true, 4, double, false> >>);
	REGISTER_KERNEL("Iso: DDA - [num] Marmitt (float, stable)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalMarmitt<true, 4, float, true> >>);
	//REGISTER_KERNEL("Iso: DDA - [num] Marmitt (double, stable)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA8<kernel::IsoVoxelEvalMarmitt<true, 4, double, true> >>);
	
	REGISTER_KERNEL("Iso: Cubic DDA - fixed step (no poly)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA64<kernel::IsoTricubicVoxelEvalFixedStepsNoPoly >>);
	REGISTER_KERNEL("Iso: Cubic DDA - fixed step (loop)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA64<kernel::IsoTricubicVoxelEvalFixedSteps<kernel::TricubicFactorAlgorithm::LOOP> >>);
	REGISTER_KERNEL("Iso: Cubic DDA - Sphere Simple (loop)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA64<kernel::IsoTricubicVoxelEvalSphereTracing<kernel::SimpleBound, kernel::TricubicFactorAlgorithm::LOOP> >>);
	REGISTER_KERNEL("Iso: Cubic DDA - Sphere Bernstein (loop)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA64<kernel::IsoTricubicVoxelEvalSphereTracing<kernel::BernsteinBound, kernel::TricubicFactorAlgorithm::LOOP> >>);
	REGISTER_KERNEL("Iso: Cubic DDA - fixed step (explicit)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA64<kernel::IsoTricubicVoxelEvalFixedSteps<kernel::TricubicFactorAlgorithm::EXPLICIT> >>);
	REGISTER_KERNEL("Iso: Cubic DDA - Sphere Simple (explicit)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA64<kernel::IsoTricubicVoxelEvalSphereTracing<kernel::SimpleBound, kernel::TricubicFactorAlgorithm::EXPLICIT> >>);
	REGISTER_KERNEL("Iso: Cubic DDA - Sphere Bernstein (explicit)", kernel::IsosurfaceKernel<kernel::IsosurfaceDDA64<kernel::IsoTricubicVoxelEvalSphereTracing<kernel::BernsteinBound, kernel::TricubicFactorAlgorithm::EXPLICIT> >>);
	
}
