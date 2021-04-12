#include <catch.hpp>
#include <json.hpp>
#include <fstream>
#include <memory>

#include <lib.h>
#include <tinyformat.h>
#include "test_utils.h"

template<class ArgType>
struct eigen_type_helper {
	typedef Eigen::Matrix<typename ArgType::Scalar,
		ArgType::SizeAtCompileTime,
		ArgType::SizeAtCompileTime,
		Eigen::ColMajor,
		ArgType::MaxSizeAtCompileTime,
		ArgType::MaxSizeAtCompileTime> MatrixType;
};

template<class ArgType>
class max_error_functor {
	const ArgType& m1, m2;
	const typename ArgType::Scalar bound;
public:
	max_error_functor(const ArgType& m1, const ArgType& m2, const typename ArgType::Scalar bound)
	: m1(m1), m2(m2), bound(bound) {}

	typename ArgType::Scalar operator() (Eigen::Index row, Eigen::Index col) const {
		bool hit1 = m1(row, col) > 0;
		bool hit2 = m2(row, col) > 0;
		auto r = (hit1 == hit2) ? std::abs(m1(row, col)-m2(row,col)) : ArgType::Scalar(0);
		//if (r > 0.2f) {
		//	std::cout << "row=" << row << ", col=" << col << ", m1=" << m1(row, col) << ", m2=" << m2(row, col) << std::endl;
		//	//__debugbreak();
		//}
		if (bound > 0)
			r = r > bound ? ArgType::Scalar(1) : ArgType::Scalar(0);
		return r;
	}
};
template <class ArgType>
Eigen::CwiseNullaryOp<max_error_functor<ArgType>, typename eigen_type_helper<ArgType>::MatrixType>
absoluteErrorWithoutWrongHits(const Eigen::MatrixBase<ArgType>& arg1, const Eigen::MatrixBase<ArgType>& arg2, float bound=-1)
{
	typedef typename eigen_type_helper<ArgType>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(arg1.rows(), arg1.cols(),
		max_error_functor<ArgType>(arg1.derived(), arg2.derived(), bound));
}

template<class ArgType>
class wrong_hit_functor {
	const ArgType& m1, m2;
public:
	wrong_hit_functor(const ArgType& m1, const ArgType& m2) : m1(m1), m2(m2) {}

	typename ArgType::Scalar operator() (Eigen::Index row, Eigen::Index col) const {
		bool hit1 = m1(row, col) > 0;
		bool hit2 = m2(row, col) > 0;
		return (hit1 == hit2) ? ArgType::Scalar(0) : ArgType::Scalar(1);
	}
};
template <class ArgType>
Eigen::CwiseNullaryOp<wrong_hit_functor<ArgType>, typename eigen_type_helper<ArgType>::MatrixType>
wrongHits(const Eigen::MatrixBase<ArgType>& arg1, const Eigen::MatrixBase<ArgType>& arg2)
{
	typedef typename eigen_type_helper<ArgType>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(arg1.rows(), arg1.cols(), 
		wrong_hit_functor<ArgType>(arg1.derived(), arg2.derived()));
}

template<class ArgType>
class relative_error_functor {
	const ArgType& m1, m2;
	const typename ArgType::Scalar bound;
public:
	relative_error_functor(const ArgType& m1, const ArgType& m2, typename ArgType::Scalar bound)
		: m1(m1), m2(m2), bound(bound) {}

	typename ArgType::Scalar operator() (Eigen::Index row, Eigen::Index col) const {
		bool hit1 = m1(row, col) > 0;
		bool hit2 = m2(row, col) > 0;
		if (hit1 && hit2)
		{
			return (std::abs(1 - m2(row, col) / m1(row, col)) > bound) ? ArgType::Scalar(1) : ArgType::Scalar(0);
		} else
		{
			return ArgType::Scalar(0);
		}
	}
};
template <class ArgType>
Eigen::CwiseNullaryOp<relative_error_functor<ArgType>, typename eigen_type_helper<ArgType>::MatrixType>
relativeError(const Eigen::MatrixBase<ArgType>& gt, const Eigen::MatrixBase<ArgType>& approx, float bound)
{
	typedef typename eigen_type_helper<ArgType>::MatrixType MatrixType;
	return MatrixType::NullaryExpr(gt.rows(), gt.cols(),
		relative_error_functor<ArgType>(gt.derived(), approx.derived(), bound));
}

TEST_CASE("Iso-Precision", "[Iso][!hide]")
{
	static const std::vector<std::string> SETTINGS_FILES = {
		"../tests/ejecta-close.json",
		"../tests/ejecta-far.json" };
	static const std::string GT_KERNEL = "Iso: Fixed step size - trilinear";
	static const float GT_STEPSIZE = 0.001f;
	//(kernel name, stepsize, test iff same as GT)
	static const std::vector<std::tuple<std::string, float, bool>> TEST_CASES = {
		{"Iso: Fixed step size - trilinear", 0.002f, true},
		{"Iso: Fixed step size - trilinear", 0.05f, false},
		{"Iso: Fixed step size - trilinear", 0.10f, false},
		{"Iso: Fixed step size - trilinear", 0.20f, false},
		{"Iso: Fixed step size - trilinear", 0.50f, false},

		{"Iso: DDA - fixed step", 0.05f, true},
		{"Iso: DDA - [ana] hyperbolic (float)", 1.0f, false},
		{"Iso: DDA - [ana] Schwarze (float)", 1.0f, false},
		{"Iso: DDA - [num] midpoint", 1.0f, false},
		{"Iso: DDA - [num] linear", 1.0f, false},
		{"Iso: DDA - [num] Neubauer", 1.0f, false},
		{"Iso: DDA - [num] Marmitt (float, unstable)", 1.0f, false},
		{"Iso: DDA - [num] Marmitt (float, stable)", 1.0f, true},
	};
	static const float ERROR_BOUND = 0.0001f;
	static const std::string FORMAT_HEADER1 = tinyformat::format(
		" %40s %5s | %11s %11s %11s %11s %11s %11s",
		"Kernel", "step", "avg error", "max error", "wrong hits", ">0.1% error", ">0.5% error", ">2% error"
		);
	static const std::string FORMAT_HEADER2(FORMAT_HEADER1.size(), '-');
	static const char* FORMAT = " %40s %5.3f | %11.7f %11.7f %10.5f%% %10.5f%% %10.5f%% %10.5f%%";
	static int RESOLUTION_X = 512;
	static int RESOLUTION_Y = 512;

	static const std::vector<std::string> STATISTICS_FILES = {
		"../results/statistics/iso-ejecta/IsoPrecisionClose.txt",
		"../results/statistics/iso-ejecta/IsoPrecisionFar.txt"
	};
	static const bool WRITE_STATISTICS = true;
	
	//allocate output
	renderer::OutputTensor gtOutput(
		RESOLUTION_Y, RESOLUTION_X, renderer::IsoRendererOutputChannels);
	renderer::OutputTensor output(
		RESOLUTION_Y, RESOLUTION_X, renderer::IsoRendererOutputChannels);
	
	//load kernels
	bool kernelsCompiledSuccessfully = renderer::KernelLauncher::Instance().init();
	//not needed, init() already compiles the kernels
	//bool kernelsCompiledSuccessfully = renderer::KernelLauncher::Instance().reload(std::cout, true);
	REQUIRE(kernelsCompiledSuccessfully);
	auto isoKernels = renderer::KernelLauncher::Instance().getKernelNames(
		renderer::KernelLauncher::KernelTypes::Iso);

	for (size_t idx=0; idx<SETTINGS_FILES.size(); ++idx)
	{
		const auto& settingsFile = SETTINGS_FILES[idx];
		
		//prepare statistics output
		std::ofstream statisticsFile;
		if (WRITE_STATISTICS) {
			statisticsFile.open(STATISTICS_FILES[idx]);
			statisticsFile << "# Kernel,\tstepsize,\tdepth values. Tab-separated\n";
		}
		
		//load settings
		std::ifstream i(settingsFile);
		REQUIRE(i.is_open());
		nlohmann::json settings;
		i >> settings;
		i.close();

		renderer::RendererArgs rendererArgs;
		renderer::Camera camera;
		std::string volumeFileName;
		renderer::RendererArgs::load(settings, { "./" }, rendererArgs, camera, volumeFileName);
		rendererArgs.cameraResolutionX = RESOLUTION_X;
		rendererArgs.cameraResolutionY = RESOLUTION_Y;
		static const int DEPTH_CHANNEL = 4;

		//load volume
		auto volume = std::make_shared<renderer::Volume>(volumeFileName);
		if (rendererArgs.mipmapLevel != 0)
			volume->createMipmapLevel(rendererArgs.mipmapLevel, renderer::Volume::MipmapFilterMode::AVERAGE);
		volume->getLevel(rendererArgs.mipmapLevel)->copyCpuToGpu();

		//update render arguments
		camera.updateRenderArgs(rendererArgs);
		rendererArgs.shading.lightDirection = camera.screenToWorld(rendererArgs.shading.lightDirection);

		//render ground truth
		REQUIRE(std::find(isoKernels.begin(), isoKernels.end(), GT_KERNEL) != isoKernels.end());
		rendererArgs.stepsize = GT_STEPSIZE;
		cudaStream_t stream = cuMat::Context::current().stream();
		renderer::render_gpu(GT_KERNEL, volume.get(), &rendererArgs, gtOutput, stream);
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		auto gtDepth = extractChannel(gtOutput, DEPTH_CHANNEL);
		if (WRITE_STATISTICS)
		{
			statisticsFile << "\"" << GT_KERNEL << "\"\t" << GT_STEPSIZE;
			for (size_t x = 0; x < gtDepth.rows(); ++x)
				for (size_t y = 0; y < gtDepth.cols(); ++y)
					statisticsFile << "\t" << gtDepth(x, y);
			statisticsFile << std::endl;
		}

		//run test cases
		std::cout << "\nRun " << settingsFile << std::endl;
		std::cout << FORMAT_HEADER1 << "\n" << FORMAT_HEADER2 << std::endl;
		std::vector<std::tuple<std::string, float, float, float>> to_validate;
		for (const auto& testcase : TEST_CASES)
		{
			const std::string& kernelName = std::get<0>(testcase);
			const float stepsize = std::get<1>(testcase);
			const bool validate = std::get<2>(testcase);

			rendererArgs.stepsize = stepsize;
			renderer::render_gpu(kernelName, volume.get(), &rendererArgs, output, stream);
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
			auto depth = extractChannel(output, DEPTH_CHANNEL);

			auto difference = (gtDepth - depth).cwiseAbs();
			float avgError = difference.mean();
			float maxError = absoluteErrorWithoutWrongHits(gtDepth, depth).maxCoeff();
			float wrongHitsPercentage = wrongHits(gtDepth, depth).sum() / (depth.rows() * depth.cols());
#if 0
			//relative error
			float quantile01 = relativeError(gtDepth, depth, 0.1 / 100.0f).sum() / (depth.rows() * depth.cols());
			float quantile05 = relativeError(gtDepth, depth, 0.5 / 100.0f).sum() / (depth.rows() * depth.cols());
			float quantile20 = relativeError(gtDepth, depth, 2.0 / 100.0f).sum() / (depth.rows() * depth.cols());
#else
			//absolute error (depth is an absolute quantity)
			float quantile01 = absoluteErrorWithoutWrongHits(gtDepth, depth, 0.1 / 100.0f).sum() / (depth.rows() * depth.cols());
			float quantile05 = absoluteErrorWithoutWrongHits(gtDepth, depth, 0.5 / 100.0f).sum() / (depth.rows() * depth.cols());
			float quantile20 = absoluteErrorWithoutWrongHits(gtDepth, depth, 2.0 / 100.0f).sum() / (depth.rows() * depth.cols());
#endif
			std::cout << tinyformat::format(FORMAT,
				kernelName, stepsize, avgError, maxError,
				wrongHitsPercentage * 100, quantile01 * 100, quantile05 * 100, quantile20 * 100)
				<< std::endl;

			if (validate) {
				to_validate.emplace_back(kernelName, stepsize, avgError, wrongHitsPercentage);
			}

			if (WRITE_STATISTICS)
			{
				statisticsFile << "\"" << kernelName << "\"\t" << stepsize;
				for (size_t x = 0; x < depth.rows(); ++x)
					for (size_t y = 0; y < depth.cols(); ++y)
						statisticsFile << "\t" << depth(x, y);
				statisticsFile << std::endl;
			}
		}
		INFO("Settings: " << settingsFile);
		for (const auto& test : to_validate)
		{
			const std::string& kernelName = std::get<0>(test);
			const float stepsize = std::get<1>(test);
			const float avgError = std::get<2>(test);
			const float wrongHits = std::get<3>(test);
			INFO(kernelName << " - " << stepsize);
			CHECK(avgError < ERROR_BOUND);
			CHECK(wrongHits < ERROR_BOUND);
		}
	}
}