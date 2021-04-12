#include "test_utils.h"

Eigen::MatrixXf extractChannel(const renderer::OutputTensor& output, int channel)
{
	return output.slice(channel).eval().toEigen();
}

__constant__ float TestConstantMemory[16];

__global__ void TestCopyConstantToMemory(dim3 virtual_size, float* output)
{
	CUMAT_KERNEL_1D_LOOP(i, virtual_size)
	{
		output[i] = TestConstantMemory[i];
	}
	CUMAT_KERNEL_1D_LOOP_END
}

void testConstantMemory()
{
	float data[16] = { 1,2,3,4,5,6,7,8,9,12,1 };
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(TestConstantMemory, data, sizeof(float) * 16));

	cuMat::VectorXf out(16);
	cuMat::Context& ctx = cuMat::Context::current();
	const auto cfg = ctx.createLaunchConfig1D(16, TestCopyConstantToMemory);
	TestCopyConstantToMemory
		<<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
			cfg.virtual_size, out.data());
	CUMAT_CHECK_ERROR();
	const auto outCpu = out.toEigen();
	std::cout << outCpu << std::endl;
}