#include "testTrapezoidBounds.h"

#include <catch.hpp>
#include <json.hpp>
#include <fstream>
#include <memory>
#include <map>
#include <random>
#include <cassert>
#include <stdarg.h>
#include <stdio.h>

#include <lib.h>
#include <tinyformat.h>
#include <renderer_math_cpu.h>

namespace {

	template<int N>
	struct GetFromFloat3;
	template<> struct GetFromFloat3<1>
	{
		float operator()(const float3& v) const { return v.x; }
		double operator()(const double3& v) const { return v.x; }
	};
	template<> struct GetFromFloat3<2>
	{
		float operator()(const float3& v) const { return v.y; }
		double operator()(const double3& v) const { return v.y; }
	};
	template<> struct GetFromFloat3<3>
	{
		float operator()(const float3& v) const { return v.z; }
		double operator()(const double3& v) const { return v.z; }
	};
	template<int N, int Q2, int P2, int Q3>
	float getMax(float a, float b, 
		const kernel::PolyExpPoly<Q2, float3, P2, float>& q2, 
		const kernel::Polynomial<Q3, float3>& q3)
	{
		const auto& roots = RendererMathCpu::extractRealRoots(
			RendererMathCpu::roots(q3.cast(GetFromFloat3<N>())));
		float max = 0;
		const auto q2Part = kernel::PolyExpPoly<Q2, float, P2, float>(q2.q.cast(GetFromFloat3<N>()), q2.p);
		for (double x : roots)
		{
			if (x > a && x < b)
				max = std::max(max, q2Part(float(x)));
		}
		return max;
	}
	template<int N, int Q2, int P2, int Q3>
	double getMax(float a, float b,
		const kernel::PolyExpPoly<Q2, double3, P2, double>& q2,
		const kernel::Polynomial<Q3, double3>& q3)
	{
		const auto& roots = RendererMathCpu::extractRealRoots(
			RendererMathCpu::roots(q3.cast(GetFromFloat3<N>())));
		double max = 0;
		const auto q2Part = kernel::PolyExpPoly<Q2, double, P2, double>(q2.q.cast(GetFromFloat3<N>()), q2.p);
		for (double x : roots)
		{
			if (x > a && x < b)
				max = std::max(max, q2Part(x));
		}
		return max;
	}

	class Stats
	{
		std::vector<float> data;
		double mean_;
	public:
		void append(float f) { data.push_back(f); }
		void finish()
		{
			std::sort(data.begin(), data.end());
			double h = 1.0 / data.size();
			mean_ = 0;
			for (auto i = 0u; i < data.size(); ++i)
				mean_ += h * data[i];
		}
		float mean() const
		{
			return mean_;
		}
		float median() const { return data[data.size() / 2]; }
		float min() const { return data[0]; }
		float max() const { return data[data.size() - 1]; }
		float quantile(float q) const { return data[q * data.size()]; }
	};

	inline float max3(const float3& v) { return std::max({ v.x, v.y, v.z }); }
	inline double max3(const double3& v) { return std::max({ v.x, v.y, v.z }); }

	struct ToDouble3
	{
		double3 operator()(const float3& v) const { return make_double3(v); }
	};
	struct ToDouble
	{
		double operator()(const float& v) const { return v; }
	};

	enum class IntegrationMethod
	{
		RECTANGLE,
		TRAPEZOID,
		SIMPSON
	};
	template<IntegrationMethod Method, typename Scalar_t, int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	std::vector<Scalar_t> numericalErrorBound(
		const kernel::PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>& pep,
		const Scalar_t& a, const Scalar_t& b,
		const Scalar_t& convergence,
		int minN, int stepN,
		int maxN = 100)
	{
		using Result_t = typename kernel::PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>::Coeff_t;
		std::vector<Result_t> values;
		//refine until convergence
		Result_t lastValue {0};
		for (int i = minN; i <= maxN; i += stepN)
		{
			Result_t value;
			switch (Method)
			{
			case IntegrationMethod::RECTANGLE:
				value = pep.integrateRectangle(a, b, i);
				break;
			case IntegrationMethod::TRAPEZOID:
				value = pep.integrateTrapezoid(a, b, i);
				break;
			case IntegrationMethod::SIMPSON:
				value = pep.integrateSimpson(a, b, i);
				break;
			}
			values.push_back(value);
			if (i>0 && max3(fabs(value - lastValue)) < convergence)
				break; //early convergence
			lastValue = value;
		}

		//compute error
		std::vector<Scalar_t> errors(values.size());
		lastValue = *values.rbegin();
		for (size_t i = 0; i < values.size(); ++i)
			errors[i] = max3(fabs(values[i] - lastValue));
		return errors;
	}

	template<typename Scalar_t>
	int findNForError(const std::vector<Scalar_t>& errors, Scalar_t bound, int minN, int stepN)
	{
		int i;
		for (i = 0; i < static_cast<int>(errors.size()); ++i)
			if (errors[i] < bound)
				break;
		return i * stepN + minN;
	}

	int ffprintf(FILE* fp1, FILE* fp2, char const* fmt, ...)
	{
		va_list args;
		va_start(args, fmt);
		int rc1 = vfprintf(fp1, fmt, args);
		va_end(args);
		va_start(args, fmt);
		int rc2 = vfprintf(fp2, fmt, args);
		va_end(args);
		assert(rc1 == rc2);
		return rc1;
	}
}

static void printProgress(const std::string& prefix, float progress)
{
	int barWidth = 50;
	std::cout << prefix << " [";
	int pos = static_cast<int>(barWidth * progress);
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << std::setprecision(3) << progress * 100.0 << " %     \r";
	std::cout.flush();
	if (progress >= 1) std::cout << std::endl;
}


void testIntegrationBounds()
{
	const std::string SETTINGS_FILE = "../scenes/dvrLobb-scene2.json";
	//const std::string SETTINGS_FILE = "../scenes/dvrLobb-scene3.json";
	const std::string ROOT_PATH = "..";
	const int RESOLUTION = 512;
	const int VOLUME_RESOLUTION = 64;
	const std::string OUTPUT_FILE = "../results/statistics/dvr-marschner-lobb/trapezoid-bounds.txt";
	const std::string OUTPUT_FILE2 = "../results/statistics/dvr-marschner-lobb/integration-bounds.txt";

	//allocate output
	renderer::OutputTensor outputCumat(
		RESOLUTION, RESOLUTION, renderer::DvrRendererOutputChannels);
	kernel::OutputTensor outputRenderer{
		outputCumat.data(), outputCumat.rows(), outputCumat.cols(), outputCumat.batches() };

	//load settings
	std::ifstream i(SETTINGS_FILE);
	assert(i.is_open());
	nlohmann::json settingsJson;
	i >> settingsJson;
	i.close();

	renderer::RendererArgs rendererArgs;
	renderer::Camera camera;
	std::string volumeFileName;
	renderer::RendererArgs::load(settingsJson, { ROOT_PATH }, rendererArgs, camera, volumeFileName);
	rendererArgs.cameraResolutionX = RESOLUTION;
	rendererArgs.cameraResolutionY = RESOLUTION;

	//create marschner lobb
	auto volume = renderer::Volume::createImplicitDataset(
		VOLUME_RESOLUTION, renderer::Volume::ImplicitEquation::MARSCHNER_LOBB);
	volume->getLevel(0)->copyCpuToGpu();

	//convert settings
	kernel::RendererDeviceSettings settings;
	render_convert_settings(rendererArgs, volume.get(), settings);

	const std::vector<kernel::PolyWithBounds> polynomials =
		launchTrapezoidKernel(RESOLUTION, settings,
			volume->getLevel(0)->dataTexNearestGpu(),
			volume->getLevel(0)->dataTexLinearGpu(), outputRenderer);

#if 1
	std::map<std::string, Stats> stats;
	//size_t maxPolys = std::min(size_t(100), polynomials.size());
	size_t maxPolys = polynomials.size();
	std::cout << "Process " << maxPolys << " polynomials" << std::endl;
	for (size_t i = 0; i < maxPolys; ++i)
	{
		if (i%500==0)
			printProgress("Compute Statistics", i / float(maxPolys));
		const auto& pep = polynomials[i].polynomial.cast<ToDouble3, ToDouble>();
		const double a = polynomials[i].tMin;
		const double b = polynomials[i].tMax;

		const auto p2 = pep.d2();
		const auto p2expand = p2.expand();
		const auto p4 = p2expand.d2();
		const auto p4expand = p4.expand();

		//Beta2 - trapezoid
		
		//coarse bounds
		auto beta2Coarse1 = max3(p2.upperBound(a, b));
		using Scalar_t = decltype(beta2Coarse1);
		auto beta2Coarse2 = max3(p2expand.upperBound(a, b));
		stats["(T0a) beta2 - coarse 1"].append(beta2Coarse1);
		stats["(T0b) beta2 - coarse 2"].append(beta2Coarse2);

		//fine bounds
		const auto p3expand = p2expand.d1().expand();
		auto beta2Fine = std::max({
			max3(p2(a)), max3(p2(b)),
			getMax<1>(a, b, p2expand, p3expand.q),
			getMax<2>(a, b, p2expand, p3expand.q) ,
			getMax<3>(a, b, p2expand, p3expand.q) });
		stats["(T0c) beta2 - fine"].append(beta2Fine);

		//bernstein bounds
		auto beta2BernsteinRecursive = max3(pep.derivativeBoundsBernstein<2>(a, b));
		auto beta2BernsteinExpand = max3(p2expand.absBoundBernstein(a, b));

		//sample count
		const auto trapezoidN = [a, b](Scalar_t beta, Scalar_t error)
		{
			return 1+std::ceil((b - a) / std::sqrt(12.0 * error / ((b - a) * beta)));
		};
		stats["(T2a) N - e=1e-2 - coarse 1"].append(trapezoidN(beta2Coarse1, 1e-2));
		stats["(T2b) N - e=1e-2 - coarse 2"].append(trapezoidN(beta2Coarse2, 1e-2));
		stats["(T2c) N - e=1e-2 - fine"].append(trapezoidN(beta2Fine, 1e-2));
		stats["(T2d) N - e=1e-2 - bs rec"].append(trapezoidN(beta2BernsteinRecursive, 1e-2));
		stats["(T2e) N - e=1e-2 - bs expand"].append(trapezoidN(beta2BernsteinExpand, 1e-2));
		
		stats["(T3a) N - e=1e-3 - coarse 1"].append(trapezoidN(beta2Coarse1, 1e-3));
		stats["(T3b) N - e=1e-3 - coarse 2"].append(trapezoidN(beta2Coarse2, 1e-3));
		stats["(T3c) N - e=1e-3 - fine"].append(trapezoidN(beta2Fine, 1e-3));
		stats["(T3d) N - e=1e-3 - bs rec"].append(trapezoidN(beta2BernsteinRecursive, 1e-3));
		stats["(T3e) N - e=1e-3 - bs expand"].append(trapezoidN(beta2BernsteinExpand, 1e-3));
		
		stats["(T5a) N - e=1e-5 - coarse 1"].append(trapezoidN(beta2Coarse1, 1e-5));
		stats["(T5b) N - e=1e-5 - coarse 2"].append(trapezoidN(beta2Coarse2, 1e-5));
		stats["(T5c) N - e=1e-5 - fine"].append(trapezoidN(beta2Fine, 1e-5));
		stats["(T5d) N - e=1e-5 - bs rec"].append(trapezoidN(beta2BernsteinRecursive, 1e-5));
		stats["(T5e) N - e=1e-5 - bs expand"].append(trapezoidN(beta2BernsteinExpand, 1e-5));
		
		stats["(T9a) N - e=1e-9 - coarse 1"].append(trapezoidN(beta2Coarse1, 1e-9));
		stats["(T9b) N - e=1e-9 - coarse 2"].append(trapezoidN(beta2Coarse2, 1e-9));
		stats["(T9c) N - e=1e-9 - fine"].append(trapezoidN(beta2Fine, 1e-9));
		stats["(T9d) N - e=1e-9 - bs rec"].append(trapezoidN(beta2BernsteinRecursive, 1e-9));
		stats["(T9e) N - e=1e-9 - bs expand"].append(trapezoidN(beta2BernsteinExpand, 1e-9));

		//numerically
		const auto errorsTrapezoid = numericalErrorBound<IntegrationMethod::TRAPEZOID, Scalar_t>(pep, a, b, 1e-11, 1, 1, 1000);
		stats["(T2f) N - e=1e-2 - numerically"].append(findNForError(errorsTrapezoid, 1e-2, 1, 1));
		stats["(T3f) N - e=1e-3 - numerically"].append(findNForError(errorsTrapezoid, 1e-3, 1, 1));
		stats["(T5f) N - e=1e-5 - numerically"].append(findNForError(errorsTrapezoid, 1e-5, 1, 1));
		stats["(T9f) N - e=1e-9 - numerically"].append(findNForError(errorsTrapezoid, 1e-9, 1, 1));
		
		//Beta4 - simpson

		//coarse bounds
		auto beta4Coarse1 = max3(p4.upperBound(a, b));
		auto beta4Coarse2 = max3(p4expand.upperBound(a, b));
		stats["(s0a) beta4 - coarse 1"].append(beta4Coarse1);
		stats["(s0b) beta4 - coarse 2"].append(beta4Coarse2);

		//fine bounds
		const auto p5expand = p4expand.d1().expand();
		auto beta4Fine = std::max({
			max3(p4(a)), max3(p4(b)),
			getMax<1>(a, b, p4expand, p5expand.q),
			getMax<2>(a, b, p4expand, p5expand.q) ,
			getMax<3>(a, b, p4expand, p5expand.q) });
		stats["(s0c) beta4 - fine"].append(beta4Fine);

		//bernstein bounds
		auto beta4BernsteinRecursive = max3(pep.derivativeBoundsBernstein<4>(a, b));
		auto beta4BernsteinExpand = max3(p4expand.absBoundBernstein(a, b));

		//adaptive simpson
		const auto adaptiveSimpsonN = [a, b, pep](Scalar_t error)
		{
			int N = 0;
			auto h = (b - a);
			kernel::Quadrature::adaptiveSimpson(
				pep, a, b, h, error, N, 1e-10);
			return N;
		};
		
		//sample count
		const auto simpsonN = [a, b](Scalar_t beta, Scalar_t error)
		{
			int N = std::ceil((b - a) / std::sqrt(std::sqrt(180.0 * error / ((b - a) * (beta+1e-7)))));
			if ((N & 1) == 1) N++; //round-up to next even number
			return N+1;
		};
		stats["(s2a) N - e=1e-2 - coarse 1"].append(simpsonN(beta4Coarse1, 1e-2));
		stats["(s2b) N - e=1e-2 - coarse 2"].append(simpsonN(beta4Coarse2, 1e-2));
		stats["(s2c) N - e=1e-2 - fine"].append(simpsonN(beta4Fine, 1e-2));
		stats["(s2d) N - e=1e-2 - bs rec"].append(simpsonN(beta4BernsteinRecursive, 1e-2));
		stats["(s2e) N - e=1e-2 - bs expand"].append(simpsonN(beta4BernsteinExpand, 1e-2));
		stats["(s2f) N - e=1e-2 - adaptive"].append(adaptiveSimpsonN(1e-2));
		
		stats["(s3a) N - e=1e-3 - coarse 1"].append(simpsonN(beta4Coarse1, 1e-3));
		stats["(s3b) N - e=1e-3 - coarse 2"].append(simpsonN(beta4Coarse2, 1e-3));
		stats["(s3c) N - e=1e-3 - fine"].append(simpsonN(beta4Fine, 1e-3));
		stats["(s3d) N - e=1e-3 - bs rec"].append(simpsonN(beta4BernsteinRecursive, 1e-3));
		stats["(s3e) N - e=1e-3 - bs expand"].append(simpsonN(beta4BernsteinExpand, 1e-3));
		stats["(s3f) N - e=1e-3 - adaptive"].append(adaptiveSimpsonN(1e-3));
		
		stats["(s5a) N - e=1e-5 - coarse 1"].append(simpsonN(beta4Coarse1, 1e-5));
		stats["(s5b) N - e=1e-5 - coarse 2"].append(simpsonN(beta4Coarse2, 1e-5));
		stats["(s5c) N - e=1e-5 - fine"].append(simpsonN(beta4Fine, 1e-5));
		stats["(s5d) N - e=1e-5 - bs rec"].append(simpsonN(beta4BernsteinRecursive, 1e-5));
		stats["(s5e) N - e=1e-5 - bs expand"].append(simpsonN(beta4BernsteinExpand, 1e-5));
		stats["(s5f) N - e=1e-5 - adaptive"].append(adaptiveSimpsonN(1e-5));
		
		stats["(s9a) N - e=1e-9 - coarse 1"].append(simpsonN(beta4Coarse1, 1e-9));
		stats["(s9b) N - e=1e-9 - coarse 2"].append(simpsonN(beta4Coarse2, 1e-9));
		stats["(s9c) N - e=1e-9 - fine"].append(simpsonN(beta4Fine, 1e-9));
		stats["(s9d) N - e=1e-9 - bs rec"].append(simpsonN(beta4BernsteinRecursive, 1e-9));
		stats["(s9e) N - e=1e-9 - bs expand"].append(simpsonN(beta4BernsteinExpand, 1e-9));
		stats["(s9f) N - e=1e-9 - adaptive"].append(adaptiveSimpsonN(1e-9));
		
		const auto errorsSimpson = numericalErrorBound<IntegrationMethod::SIMPSON, Scalar_t>(pep, a, b, 1e-11, 2, 2, 500);
		stats["(s2g) N - e=1e-2 - numerically"].append(findNForError(errorsSimpson, 1e-2, 2, 2));
		stats["(s3g) N - e=1e-3 - numerically"].append(findNForError(errorsSimpson, 1e-3, 2, 2));
		stats["(s5g) N - e=1e-5 - numerically"].append(findNForError(errorsSimpson, 1e-5, 2, 2));
		stats["(s9g) N - e=1e-9 - numerically"].append(findNForError(errorsSimpson, 1e-9, 2, 2));
	}
	printProgress("Compute Statistics", 1.0f);

	FILE* ofp = fopen(OUTPUT_FILE2.c_str(), "w");
	ffprintf(stdout, ofp, "%% Required function evaluations\n");
	ffprintf(stdout, ofp, "         Statistic             |    min    |    max    |    mean   |   median  |    75%%    |    90%%    |\n");
	const auto digits = [](float v) -> int {return std::max(0, std::min(5, 6 - int(std::log10(v)))); };
	for (auto& e : stats)
	{
		e.second.finish();
		ffprintf(stdout, ofp, "%-30s | %9.*f | %9.*f | %9.*f | %9.*f | %9.*f | %9.*f |\n",
			e.first.c_str(),
			digits(e.second.min()), e.second.min(),
			digits(e.second.max()), e.second.max(),
			digits(e.second.mean()), e.second.mean(),
			digits(e.second.median()), e.second.median(),
			digits(e.second.quantile(0.75f)), e.second.quantile(0.75f),
			digits(e.second.quantile(0.90f)), e.second.quantile(0.90f));
	}
	fclose(ofp);
	
#else
	std::cout << "Process " << polynomials.size() << " polynomials" << std::endl;
	FILE* f = fopen(OUTPUT_FILE.c_str(), "w");
	printf("  IDX    |    beta    | N for r=1e-2 | N for r=1e-3 | N for r=1e-4\n");
	fprintf(f, "  IDX    |    beta    | N for r=1e-2 | N for r=1e-3 | N for r=1e-4\n");
	for (size_t i=0; i<polynomials.size(); ++i)
	{
		const auto& pep = polynomials[i].polynomial;
		const float a = polynomials[i].tMin;
		const float b = polynomials[i].tMax;

		//find upper bound
		const auto& pep2 = pep.d2().expand();
		const auto& pep3 = pep2.d1().expand();
		float beta = 0;
		beta = std::max({ beta, pep2(a).x, pep2(a).y, pep2(a).z });
		beta = std::max({ beta, pep2(b).x, pep2(b).y, pep2(b).z });
		beta = std::max(beta, getMax<1>(a, b, pep2, pep3.q));
		beta = std::max(beta, getMax<2>(a, b, pep2, pep3.q));
		beta = std::max(beta, getMax<3>(a, b, pep2, pep3.q));

		int N2 = std::ceil((b - a) / std::sqrt(12 * 1e-2 / ((b - a) * beta)));
		int N3 = std::ceil((b - a) / std::sqrt(12 * 1e-3 / ((b - a) * beta)));
		int N4 = std::ceil((b - a) / std::sqrt(12 * 1e-4 / ((b - a) * beta)));
		printf(" %7d | %10.3e | %12d | %12d | %12d \n",
			int(i), beta, N2, N3, N4);
		fprintf(f, " %7d | %10.3e | %12d | %12d | %12d \n",
			int(i), beta, N2, N3, N4);
		if (i % 1000 == 0)
			fflush(f);
	}
	fclose(f);
#endif
}

TEST_CASE("Dvr-TrapezoidBounds", "[!hide][Dvr]")
{
	testIntegrationBounds();
}