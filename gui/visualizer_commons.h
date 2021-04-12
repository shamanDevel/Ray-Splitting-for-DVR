#pragma once

#include <cuMat/src/ForwardDeclarations.h>

//enum RenderMode
//{
//	IsosurfaceRendering,
//	DirectVolumeRendering
//};
typedef renderer::RendererArgs::RenderMode RenderMode;
typedef renderer::RendererArgs::DvrTfMode DvrTfMode;

typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 2, cuMat::ColumnMajor> FlowTensor;