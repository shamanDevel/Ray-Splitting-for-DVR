#pragma once

#include <cuMat/src/Matrix.h>

namespace renderer
{
	typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> OutputTensor;
}
Eigen::MatrixXf extractChannel(const renderer::OutputTensor& output, int channel);

void testConstantMemory();
