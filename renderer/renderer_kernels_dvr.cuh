#pragma once
/**
 * Kernel definitions, included either in renderer_static.cu or renderer_rtc_kernels.cu
 */

#include "helper_math.cuh"
#include "renderer_settings.cuh"
#include "renderer_utils.cuh"
#include "renderer_impl_dvr.cuh"
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

	__device__ inline void writeOutputDvr(
		OutputTensor& output, int x, int y,
		const float3& rgb, float alpha,
		const float3& normal, float depth, const float2& flow)
	{
		output.coeff(y, x, 0) = rgb.x;
		output.coeff(y, x, 1) = rgb.y;
		output.coeff(y, x, 2) = rgb.z;
		output.coeff(y, x, 3) = alpha;
		output.coeff(y, x, 4) = normal.x;
		output.coeff(y, x, 5) = normal.y;
		output.coeff(y, x, 6) = normal.z;
		output.coeff(y, x, 7) = depth;
		output.coeff(y, x, 8) = flow.x;
		output.coeff(y, x, 9) = flow.y;
	}
	__device__ inline void writeOutputDvr(
		OutputTensor& output, int x, int y)
	{
		writeOutputDvr(output, x, y, make_float3(0, 0, 0), 0, make_float3(0, 0, 0), 0, make_float2(0, 0));
	}

	/**
	 * The rendering kernel with the parallel loop over the pixels and output handling
	 */
	template<typename Functor>
	__global__ void DvrKernel(dim3 virtual_size,
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
			PerPixelInstrumentation*  instrumentation = &localInstrumentation;
			
			int x = x_ + cDeviceSettings.viewport.x;
			int y = y_ + cDeviceSettings.viewport.y;
			float posx = ((x + 0.5f) / cDeviceSettings.screenSize.x) * 2 - 1;
			float posy = ((y + 0.5f) / cDeviceSettings.screenSize.y) * 2 - 1;

			//target world position
			float4 screenPosStart = make_float4(posx, posy, 0.0, 1);
			float4 eyePos4 = matmul(cDeviceSettings.currentViewMatrixInverse, screenPosStart);
			eyePos4 /= eyePos4.w;
			float3 eyePos = make_float3(eyePos4);
			
			float4 screenPos = make_float4(posx, posy, 0.9, 1);
			float4 worldPos = matmul(cDeviceSettings.currentViewMatrixInverse, screenPos);
			worldPos /= worldPos.w;

			//ray direction
			float3 rayDir = normalize(make_float3(worldPos) - eyePos);

			//entry, exit points
			float tmin, tmax;
			intersectionRayAABB(eyePos, rayDir, cDeviceSettings.boxMin, cDeviceSettings.boxSize, tmin, tmax);
			
			//clip plane
			if (cDeviceSettings.enableClipPlane)
			{
				float3 entryPos = eyePos;
				float nom = dot(make_float3(cDeviceSettings.clipPlane), entryPos);
				float denom = dot(make_float3(cDeviceSettings.clipPlane), rayDir);
				if (fabs(denom)>1e-5)
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
				writeOutputDvr(output, x_, y_);
#if KERNEL_INSTRUMENTATION==1
				if (instrumentationOut) instrumentationOut[__i] = localInstrumentation;
#endif
				continue;
			}

#ifndef KERNEL_NO_DEBUG
			//cDeviceSettings.pixelSelected = true;
			//cDeviceSettings.selectedPixel = make_int2(924, 101); //manually selecting the debug pixel
			
			bool debug = cDeviceSettings.pixelSelected && cDeviceSettings.selectedPixel.x == x && cDeviceSettings.selectedPixel.y == y;
			if (debug)
			{
				printf("Shoot ray, eye=(%.4f, %.4f, %.4f), dir=(%.4f, %.4f, %.4f), tmin=%.4f, tmax=%.4f\n",
					cDeviceSettings.eyePos.x, cDeviceSettings.eyePos.y, cDeviceSettings.eyePos.z,
					rayDir.x, rayDir.y, rayDir.z,
					tmin, tmax);
			}
#else
			bool debug = false;
#endif
			
			const RaytraceDvrOutput out = Functor::call(
				x, y,
				eyePos, rayDir, tmin, tmax, debug, instrumentation);

			//convert color
			float3 colorXYZ = make_float3(out.color.x, out.color.y, out.color.z);
			float3 colorRGB = xyzToRgb(colorXYZ);
			colorRGB = clamp(colorRGB, 0, 1); //TODO: tone mapping

			//evaluate flow and depth
			float depth = out.alpha > 1e-5 ? out.depth / out.alpha : 0;
			float3 posWorld = cDeviceSettings.eyePos + depth * rayDir;
			float4 screenCurrent = matmul(cDeviceSettings.currentViewMatrix, make_float4(posWorld, 1.0));
			screenCurrent /= screenCurrent.w;
			float4 screenNext = matmul(cDeviceSettings.nextViewMatrix, make_float4(posWorld, 1.0));
			screenNext /= screenNext.w;
			float2 flow = out.alpha > 1e-5
				? 0.5f*make_float2(screenCurrent.x - screenNext.x, screenCurrent.y - screenNext.y)
				: make_float2(0, 0);
			float depthScreen = out.alpha > 1e-5 ? screenCurrent.z : 0.0f;
			//evaluate normal
			float3 normalWorld = safeNormalize(out.normalWorld);
			float3 normalScreen = make_float3(matmul(cDeviceSettings.normalMatrix, make_float4(-normalWorld, 0)));

			writeOutputDvr(output, x_, y_,
				colorRGB, out.alpha, normalScreen, depthScreen, flow);
#if KERNEL_INSTRUMENTATION==1
			if (instrumentationOut) instrumentationOut[__i] = localInstrumentation;
#endif
		}
	}

	REGISTER_KERNEL("DVR: Fixed step size - nearest", kernel::DvrKernel<kernel::DvrFixedStepSizeTexture<kernel::NEAREST>>);
	REGISTER_KERNEL("DVR: Fixed step size - trilinear", kernel::DvrKernel<kernel::DvrFixedStepSizeTexture<kernel::TRILINEAR>>);
	REGISTER_KERNEL("DVR: Fixed step size - tricubic",  kernel::DvrKernel<kernel::DvrFixedStepSizeTexture<kernel::TRICUBIC>>);
	REGISTER_KERNEL("DVR: Fixed step size - trilinear (control points)", kernel::DvrKernel<kernel::DvrFixedStepSizeControlPoints<kernel::TRILINEAR>>);
	REGISTER_KERNEL("DVR: Fixed step size - trilinear (control points) - doubles", kernel::DvrKernel<kernel::DvrFixedStepSizeControlPoints_Doubles<kernel::TRILINEAR>>);
	
	//Reducing Artifacts in Volume Rendering by Higher Order Integration, deprecated
	//REGISTER_KERNEL("DVR: Simpson 1", kernel::DvrKernel<kernel::DvrSimpson1<kernel::TRILINEAR>, kernel::LinearSampledTexture >);
	
	//Controlled Precision Volume Integration
	REGISTER_KERNEL("DVR: DDA - fixed step (tf texture)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalFixedStepsTexture>>);
	REGISTER_KERNEL("DVR: DDA - fixed step (control points)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalFixedStepsControlPoints>>);
	REGISTER_KERNEL("DVR: DDA - interval simple", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimple, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval stepping (3)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorStepping<3>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval trapezoid (2)",  kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorTrapezoid<2>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval trapezoid (4)",  kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorTrapezoid<4>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval trapezoid (10)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorTrapezoid<10>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval trapezoid var", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorTrapezoid<-1>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson (2)",  kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimpson<2>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson (4)",  kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimpson<4>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson (10)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimpson<10>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson var", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimpson<-1>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson adapt", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorAdaptiveSimpson, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson shaded (2)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimpsonWithShading<2>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson shaded (10)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimpsonWithShading<10>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval Simpson shaded", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorAdaptiveSimpsonWithShading, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval PowerSeries (2)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorPowerSeries<2>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval PowerSeries (4)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorPowerSeries<4>, 15>>>);
	REGISTER_KERNEL("DVR: DDA - interval PowerSeries (10)", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorPowerSeries<10>, 15>>>);

	REGISTER_KERNEL("DVR: Fixed step size - preintegrate 1D", kernel::DvrKernel<kernel::DvrFixedStepSizePreintegrate<kernel::TRILINEAR, 1>>);
	REGISTER_KERNEL("DVR: Fixed step size - preintegrate 2D", kernel::DvrKernel<kernel::DvrFixedStepSizePreintegrate<kernel::TRILINEAR, 2>>);
	
	//Adaptive Simpson (does not work)
	//REGISTER_KERNEL("DVR: Simpson adaptive - trilinear", kernel::DvrKernel<kernel::DvrAdaptiveStepSize<kernel::TRILINEAR, true>, kernel::LinearSampledTexture >);
	
	// Multi-Iso
	REGISTER_KERNEL("MultiIso: Marmitt (3)", kernel::DvrKernel<kernel::DvrDDA<kernel::MultiIsoVoxelEval<3>>>);
	REGISTER_KERNEL("MultiIso: Marmitt (10)", kernel::DvrKernel<kernel::DvrDDA<kernel::MultiIsoVoxelEval<10>>>);
	REGISTER_KERNEL("MultiIso: fixed step - trilinear", kernel::DvrKernel<kernel::MultiIsoFixedStepSize<kernel::TRILINEAR>>);
	REGISTER_KERNEL("MultiIso: fixed step - tricubic", kernel::DvrKernel<kernel::MultiIsoFixedStepSize<kernel::TRICUBIC>>);
	
	// Hybrid
	REGISTER_KERNEL("Hybrid - Simpson (10)", kernel::DvrKernel<kernel::DvrDDA<kernel::HybridVoxelEval<kernel::DvrIntervalEvaluatorSimpson<10>, 15>>>);
	
	// Scale-invariant DVR
	REGISTER_KERNEL("DVR: Scale invariant - trilinear", kernel::DvrKernel<kernel::ScaleInvariantDvrFixedStepSize<kernel::TRILINEAR>>);
	REGISTER_KERNEL("DVR: Scale invariant - tricubic", kernel::DvrKernel<kernel::ScaleInvariantDvrFixedStepSize<kernel::TRICUBIC>>);
	REGISTER_KERNEL("DVR: Scale invariant - Simpson", kernel::DvrKernel<kernel::DvrDDA<kernel::DvrVoxelEvalScaleInvariant<15>>>);
}
