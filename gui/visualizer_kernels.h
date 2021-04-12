#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>

#include <settings.h>
//#include <volume.h>
#include <renderer.h>
#include "visualizer_commons.h"

//namespace renderer {
//	struct ShadingSettings;
//	typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> OutputTensor;
//}

namespace kernel
{
	/**
	 * \brief selects the output channel.
	 * \param inputTensor the input tensor of shape 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU
	 * \param outputBuffer the output texture, RGBA of uint8
	 * \param r the channel used for the red output
	 * \param g the channel used for the green output
	 * \param b the channel used for the blue output
	 * \param a the channel used for the alpha output
	 * \param scaleRGB scaling used on the RGB colors: output = in*scale + offset
	 * \param offsetRGB offset used on the RGB colors: output = in*scale + offset
	 * \param scaleA scaling used on the alpha color: output = in*scale + offset
	 * \param offsetA offset used on the alpha color: output = in*scale + offset
	 */
	void selectOutputChannel(
		const renderer::OutputTensor& inputTensor,
		GLubyte* outputBuffer,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB,
		float scaleA, float offsetA);

	//void selectOutputChannelSamples(
	//	const torch::Tensor& samplePositions,
	//	const torch::Tensor& sampleData,
	//	MeshDrawer::Vertex* outputVertices,
	//	int r, int g, int b, int a,
	//	float scaleRGB, float offsetRGB,
	//	float scaleA, float offsetA);

	/**
	 * \brief Performs the screen space shading.
	 * It assumes a tensor with the first 6 channels layouted as:
	 * mask, normalX, normalY, normalZ, depth, ao.
	 * \param inputTensor the input tensor of shape 1 * Channel * Height * Width
	 *  of type float32, residing on the GPU
	 * \param outputBuffer the output texture, RGBA of uint8
	 * \param settings
	 */
	void screenShading(
		const renderer::OutputTensor& inputTensor,
		GLubyte* outputBuffer,
		const renderer::ShadingSettings& settings);

	/**
	 * \brief  Fills the color map using tfTexture which is created
		according to control points. Filled color map is then displayed
		in TF Editor menu.
	 * \param colorMap surface object which makes it possible to modify color map via surface writes.
	 * \param tfTexture
	 * \param width width of color map
	 * \param height height of color map
	 */
	void fillColorMap(cudaSurfaceObject_t colorMap, cudaTextureObject_t tfTexture, int width, int height);

	/**
	 * \brief Performs fast inpainting
	 * \param input the rendered image
	 * \param maskChannel the channel that contains the mask
	 *  Entries with value <=0 are considered "empty".
	 * \param flowXChannel the channel that contains the flow along x
	 * \param flowYChannel the channel that contains the flow along x
	 * \return the inpainted tensor of shape Height * Width * 2
	 */
	FlowTensor inpaintFlow(
		const RENDERER_NAMESPACE::OutputTensor& input, 
		int maskChannel, int flowXChannel, int flowYChannel);

	/**
	 * \brief Warps the input image by the given flow field.
	 * \param input the input image
	 * \param flow the flow field (X,Y)
	 * \return the warped image
	 */
	RENDERER_NAMESPACE::OutputTensor warp(
		const RENDERER_NAMESPACE::OutputTensor& input,
		const FlowTensor& flow);

	/**
	 * \brief Extracts the minimal and maximal depth value from the input image
	 * \param input the input image
	 * \param depthChannel the channel number of the depth
	 * \return a pair with the minimal value in the first entry and the maximal value in the second.
	 */
	std::pair<float, float> extractMinMaxDepth(
		const RENDERER_NAMESPACE::OutputTensor& input, int depthChannel);

	/**
	 * Computes (1-v)*a + v*b
	 */
	RENDERER_NAMESPACE::OutputTensor lerp(
		const RENDERER_NAMESPACE::OutputTensor& a,
		const RENDERER_NAMESPACE::OutputTensor& b,
		float v);
}
