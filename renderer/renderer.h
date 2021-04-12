#pragma once

#include "commons.h"
#include "settings.h"
#include "volume.h"
#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include <cuMat/src/Matrix.h>

#include "renderer_settings.cuh"

BEGIN_RENDERER_NAMESPACE

/**
 * \brief Computes the ambient occlusion parameters.
 * 
 * \param samples the number of AO samples
 * \param rotations the number of random rotations of the kernel in every direction
 * \return a tuple with
 *   - a vector of size 'samples' containing the sample directions on the hemisphere
 *   - a vector of size 'rotations*rotations' containing the random rotations
 */
MY_API std::tuple<std::vector<float4>, std::vector<float4>> computeAmbientOcclusionParameters(int samples, int rotations);

/**
 * Channels:
 * 0: mask
 * 1,2,3: normal x,y,z
 * 4: depth
 * 5: ao
 * 6,7: flow x,y
 */
constexpr int IsoRendererOutputChannels = 8;
/**
 * Channels:
 * 0,1,2: rgb
 * 3: alpha
 * 4,5,6: normal x,y,z
 * 7: depth
 * 8,9: flow x,y
 */
constexpr int DvrRendererOutputChannels = 10;

typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> OutputTensor;

/**
 * \brief Converts renderer settings from RendererArgs to
 * kernel::RendererDeviceSettings.
 *
 * Not thread-save, contains state on the last transfer function
 * for optimizations.
 */
MY_API void render_convert_settings(
	const std::string& kernelName, const RendererArgs& in, const Volume* volume, kernel::RendererDeviceSettings& out);

/**
 * Renders the volume with the specified setting into the output tensor.
 * The rendering is restricted to the viewport specified in the render args.
 * 
 * The entries of the output matrix are interpreted in the following way:
 *   - row: y coordinate
 *   - column: x coordinate
 *   - batch: the channel
 * Hence, it provides the image correctly orientated if you print the output like a regular matrix.
 *  
 * Batch = Channels, see IsoRendererOutputChannels and DvrRendererOutputChannels
 * Height = args->viewport.w, Width = args->viewport.z
 * 
 * \param kernelName the name of the kernel, see KernelLauncher
 * \param volume the volume to render, must reside on the GPU
 * \param args the render arguments
 * \param output the output tensor, a float tensor on the GPU of size Height*Width*Batch.
 * \param stream the cuda stream. Common values:
 *  - 0: the global, synchronizing stream
 *  - cuMat::Context::current().stream() for syncing with cuMat
 *    (defined in <cuMat/src/Context.h>)
 *  - at::cuda::getCurrentCUDAStream() for syncing with PyTorch
 *    (defined in <ATen/cuda/CUDAContext.h>)
 * \param perPixelInstrumentation a device memory of width*height kernel::Instrumentation instances,
 *    if instrumentation is enabled
 * \param globalInstrumentation global instrumentations over the whole image (host memobery)
 */
MY_API void render_gpu(
	const std::string& kernelName,
	const Volume* volume,
	const RendererArgs* args, 
	OutputTensor& output,
	cudaStream_t stream,
	kernel::PerPixelInstrumentation* perPixelInstrumentation = nullptr,
	GlobalInstrumentation* globalInstrumentation = nullptr);

/**
 * Initializes the renderer.
 * For now this only sets the ambient occlusion sample directions.
 * Returns 1 on success, a negative value on failure
 */
MY_API int64_t initializeRenderer();


END_RENDERER_NAMESPACE